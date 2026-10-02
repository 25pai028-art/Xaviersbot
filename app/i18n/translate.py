"""Translation between English and Indian languages.

Providers (`TRANSLATION_PROVIDER`):
- `auto` (default): the answering LLM translates; AI4Bharat IndicTrans2 (local) is the backup when the
  LLM's translation fails the checks. Tested on college answers, the distilled IndicTrans2 models write
  fluently but jumble people's names and misplace years, while the LLM keeps them exact.
- `llm` or `indictrans2`: only that one. `off`: no translation (answers stay in English).

Safety: the answer is fact-checked in English first. A translation is used only if it is in the target
language, is not far longer than the original, and still contains every number, email and URL of the
English text; otherwise the next translator is tried, and finally the caller shows the English answer.
Emails and URLs are never shown to IndicTrans2 at all: sentences are cut around them.
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
from collections import OrderedDict

from app.config import get_settings
from app.i18n.languages import LANGUAGES

log = logging.getLogger(__name__)

EN_INDIC = "ai4bharat/indictrans2-en-indic-dist-200M"
INDIC_EN = "ai4bharat/indictrans2-indic-en-dist-200M"
BATCH = 16
MAX_SENTENCE_CHARS = 600
MAX_NEW_TOKENS = 256

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"https?://[^\s<>\"')\]]+")
_LINK = re.compile(r"(https?://[^\s<>\"')\]]*[^\s<>\"')\].,;:!?]|[\w.+-]+@[\w-]+(?:\.[\w-]+)+)")
_NUMBER = re.compile(r"\d+(?:[.,:/-]\d+)*")
_BULLET = re.compile(r"^(\s*(?:[-*•+]|\d+[.)])\s+)(.*)$")
_SENTENCE_END = re.compile(r"(?<=[.!?।؟])\s+(?=\S)")
# A full stop after these is not the end of a sentence ("Rs. 62,500", "Dr. Pravida", "St. Xavier's").
_ABBREVIATION = re.compile(
    r"(?<![\w'’])(?:rs|dr|mr|mrs|ms|prof|st|sr|jr|no|nos|vs|etc|e\.g|i\.e|approx|dept|govt|sem|ph|b|m)\.$", re.I)
_INITIAL = re.compile(r"(?<![\w'’])[A-Z]\.$")  # "A. C." in a name
_LETTERS = re.compile(r"[^\W\d_]")
# Digits in Indian scripts → ASCII, so "२०२६" counts as "2026" when checking a translation.
_NATIVE_DIGITS = {}
for _zero in (0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6, 0x0D66, 0x06F0, 0x0660):
    for _i in range(10):
        _NATIVE_DIGITS[_zero + _i] = str(_i)


class TranslationError(RuntimeError):
    pass


# ---------------------------------------------------------------- text handling
def _ascii_digits(text: str) -> str:
    return text.translate(_NATIVE_DIGITS)


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    for part in _SENTENCE_END.split(text):
        if out and (_ABBREVIATION.search(out[-1]) or _INITIAL.search(out[-1])):
            out[-1] += " " + part
        else:
            out.append(part)
    return [s for s in out if s.strip()]


def protected_items(text: str) -> tuple[set[str], set[str]]:
    """(emails and URLs, digit groups) that a translation must keep."""
    links = set(_EMAIL.findall(text)) | {u.rstrip(".,;:") for u in _URL.findall(text)}
    rest = _URL.sub(" ", _EMAIL.sub(" ", text))
    numbers = {re.sub(r"\D", "", n) for n in _NUMBER.findall(rest)}
    return links, {n for n in numbers if n}


def check_translation(source: str, translated: str) -> list[str]:
    """Numbers, emails and URLs of `source` that are missing from `translated`."""
    links, numbers = protected_items(source)
    t = _ascii_digits(translated)
    t_digits = {re.sub(r"\D", "", n) for n in _NUMBER.findall(t)}
    t_all_digits = re.sub(r"\D", "", t)
    missing = [l for l in links if l not in translated]
    missing += [n for n in numbers if n not in t_digits and n not in t_all_digits]
    return missing


def repair_numbers(source: str, translated: str) -> str:
    """Put back numbers the model cut short. The distilled model sometimes writes "202" for "2026"
    (a year it rarely saw); a number in the output that is the start of a missing source number is
    replaced by it. Anything else that is still wrong is caught by check_translation."""
    src_nums = [n for n in _NUMBER.findall(_URL.sub(" ", _EMAIL.sub(" ", source)))]
    out = _ascii_digits(translated)
    out_nums = set(_NUMBER.findall(out))
    missing = [n for n in src_nums if n not in out_nums]
    for wrong in sorted(out_nums - set(src_nums), key=len, reverse=True):
        match = next((n for n in missing if len(wrong) >= 2 and n.startswith(wrong)), None)
        if match:
            out = re.sub(rf"(?<![\d.,]){re.escape(wrong)}(?![\d])", match, out, count=1)
            missing.remove(match)
    return out


def _split(text: str) -> list[tuple[str, list[str]]]:
    """Lines as (prefix to keep, sentences to translate). Markdown bold is dropped; bullets are kept."""
    out = []
    for line in text.replace("\r", "").split("\n"):
        if not line.strip():
            out.append(("", []))
            continue
        m = _BULLET.match(line)
        prefix, body = (m.group(1), m.group(2)) if m else ("", line)
        body = body.replace("**", "").replace("__", "").strip()
        if body.startswith("#"):
            body = body.lstrip("#").strip()
        pieces = []
        # Very long "sentences" (tables flattened into one line) are cut so the model sees all of them.
        for s in _sentences(body):
            while len(s) > MAX_SENTENCE_CHARS:
                cut = s.rfind(" ", 0, MAX_SENTENCE_CHARS)
                cut = cut if cut > 100 else MAX_SENTENCE_CHARS
                pieces.append(s[:cut])
                s = s[cut:].lstrip()
            pieces.append(s)
        out.append((prefix, pieces))
    return out


def _join(lines: list[tuple[str, list[str]]]) -> str:
    return "\n".join(prefix + " ".join(s.strip() for s in sents) for prefix, sents in lines).strip()


# ---------------------------------------------------------------- IndicTrans2 text preparation
class _TextProcessor:
    """What IndicTrans2 expects around the model, as in AI4Bharat's IndicProcessor (MIT licence) minus its
    placeholder system, which the distilled models handle badly ("< ID1 >" leaking into answers):
    punctuation normalisation, Moses tokenisation for English, Indic normalisation and tokenisation with
    every Indic script transliterated to Devanagari (the model's shared script), and the reverse after."""

    _PUNC = [
        (re.compile(r"\r"), ""), (re.compile(r"\(\s*"), "("), (re.compile(r"\s*\)"), ")"),
        (re.compile(r"\s:\s?"), ":"), (re.compile(r"\s;\s?"), ";"), (re.compile(r"[`´‘‚’]"), "'"),
        (re.compile(r"[„“”«»]"), '"'), (re.compile(r"[–—]"), "-"), (re.compile(r" %"), "%"),
        (re.compile(r" [?!;]"), lambda m: m.group(0).strip()), (re.compile(r"[ ]{2,}"), " "),
        (re.compile(r"\) ([.!:?;,])"), r")\1"), (re.compile(r"(\d) %"), r"\1%"),
    ]
    _NO_TRANSLITERATION = {"Arab", "Aran", "Olck", "Mtei", "Latn"}

    def __init__(self) -> None:
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
        from indicnlp.tokenize import indic_detokenize, indic_tokenize
        from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
        from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer

        self._tok, self._detok, self._norm = indic_tokenize, indic_detokenize, IndicNormalizerFactory()
        self._xlit = UnicodeIndicTransliterator()
        self._en_tok, self._en_detok = MosesTokenizer(lang="en"), MosesDetokenizer(lang="en")
        self._en_norm = MosesPunctNormalizer()
        self._normalizers: dict = {}

    def _punc(self, text: str) -> str:
        for rx, rep in self._PUNC:
            text = rx.sub(rep, text)
        return text.strip()

    def preprocess(self, sentence: str, src: str, tgt: str) -> str:
        s_tag, t_tag = LANGUAGES[src].flores, LANGUAGES[tgt].flores
        text = _ascii_digits(self._punc(sentence))
        if src == "en":
            body = " ".join(self._en_tok.tokenize(self._en_norm.normalize(text), escape=False))
        else:
            norm = self._normalizers.get(src) or self._normalizers.setdefault(src, self._norm.get_normalizer(src))
            body = " ".join(self._tok.trivial_tokenize(norm.normalize(text), src))
            if s_tag.split("_")[1] not in self._NO_TRANSLITERATION:
                body = self._xlit.transliterate(body, src, "hi").replace(" ् ", "्")
        return f"{s_tag} {t_tag} {body.strip()}"

    def postprocess(self, sentence: str, tgt: str) -> str:
        if tgt == "en":
            return self._en_detok.detokenize(sentence.split(" "))
        if tgt == "ur":
            sentence = sentence.replace(" ؟", "؟").replace(" ۔", "۔").replace(" ،", "،").replace("ٮ۪", "ؠ")
        if tgt == "or":
            sentence = sentence.replace("ଯ଼", "ୟ")
        return self._detok.trivial_detokenize(self._xlit.transliterate(sentence, "hi", tgt), tgt)


# ---------------------------------------------------------------- IndicTrans2 models
class IndicTrans2:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: dict[str, tuple] = {}
        self._text: _TextProcessor | None = None

    @staticmethod
    def available() -> bool:
        """Both models are downloaded (checked without network access)."""
        try:
            from huggingface_hub import try_to_load_from_cache

            return all(isinstance(try_to_load_from_cache(m, "model.safetensors"), str) for m in (EN_INDIC, INDIC_EN))
        except Exception:
            return False

    def _model(self, name: str):
        """Model code from app.i18n.indictrans2 (fixed for transformers 5); weights from the Hugging Face cache."""
        if name not in self._models:
            from huggingface_hub import snapshot_download

            from app.i18n.indictrans2 import IndicTransForConditionalGeneration, IndicTransTokenizer

            # Only what this code needs: not the duplicate pytorch_model.bin (1 GB each) or the Hub's model code.
            files = ["*.json", "model.SRC", "model.TGT", "*.safetensors"]
            try:
                path = snapshot_download(name, allow_patterns=files, local_files_only=True)
            except Exception:
                path = snapshot_download(name, allow_patterns=files)
            tok = IndicTransTokenizer.from_pretrained(path)
            model = IndicTransForConditionalGeneration.from_pretrained(path).eval()
            model.tie_weights()
            self._models[name] = (tok, model)
            log.info("Loaded translation model %s", name)
        return self._models[name]

    def warmup(self) -> None:
        with self._lock:
            self._text = self._text or _TextProcessor()
            self._model(EN_INDIC)
            self._model(INDIC_EN)

    def translate_sentences(self, sentences: list[str], src: str, tgt: str) -> list[str]:
        """Translate sentences; emails and URLs inside them are kept out of the model and put back as is."""
        # Cut each sentence around links: ["Email ", "a@b.in", " for details."] → translate the text parts only.
        parts = [_LINK.split(s) for s in sentences]
        todo = [p for ps in parts for i, p in enumerate(ps) if i % 2 == 0 and _LETTERS.search(p)]
        with self._lock:
            self._text = self._text or _TextProcessor()
            tok, model = self._model(EN_INDIC if src == "en" else INDIC_EN)
            done: list[str] = []
            for i in range(0, len(todo), BATCH):
                batch = [self._text.preprocess(s, src, tgt) for s in todo[i:i + BATCH]]
                decoded = tok.batch_decode(_greedy_generate(model, tok(batch, max_length=256), MAX_NEW_TOKENS))
                done += [self._text.postprocess(d, tgt) for d in decoded]
        translated = iter(done)
        out = []
        for ps in parts:
            pieces = [(next(translated) if (i % 2 == 0 and _LETTERS.search(p)) else p.strip()) for i, p in enumerate(ps)]
            out.append(" ".join(p for p in pieces if p))
        return out


def _greedy_generate(model, enc: dict, max_new_tokens: int):
    """Greedy decoding with the model's own (tuple) key/value cache. `model.generate()` in transformers 5
    passes a Cache object that this older model code can't read; this loop gives the same result as
    generate(num_beams=1) and keeps the cache, so it stays fast on CPU."""
    import torch

    cfg = model.config
    start = cfg.decoder_start_token_id if cfg.decoder_start_token_id is not None else cfg.eos_token_id
    eos, pad = cfg.eos_token_id, cfg.pad_token_id
    with torch.inference_mode():
        encoder_outputs = model.get_encoder()(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        batch = enc["input_ids"].shape[0]
        generated = torch.full((batch, 1), start, dtype=torch.long)
        finished = torch.zeros(batch, dtype=torch.bool)
        past = None
        for _ in range(max_new_tokens):
            out = model(encoder_outputs=encoder_outputs, attention_mask=enc["attention_mask"],
                        decoder_input_ids=generated[:, -1:] if past is not None else generated,
                        past_key_values=past, use_cache=True, return_dict=True)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(dim=-1)
            nxt = torch.where(finished, torch.full_like(nxt, pad), nxt)
            generated = torch.cat([generated, nxt[:, None]], dim=1)
            finished |= nxt == eos
            if bool(finished.all()):
                break
    return generated


_indictrans = IndicTrans2()


# ---------------------------------------------------------------- LLM fallback
LLM_PROMPT = (
    "You are a translator. Translate the user's message from {src} into {tgt}; the output must be written "
    "in {tgt}. Do not answer it, only translate it. Keep every number, date, amount, phone number, email "
    "address, URL, person's name and course code (BCA, B.Com, M.Sc…) exactly as written. Keep line breaks "
    "and bullet markers. Output ONLY the translation."
)


# Names the small local model garbles ("Standard College", "Exier"): replaced by [[n]] before translating
# and put back after, so they stay exactly as written. Longest first.
LOCKED_NAMES = ["St. Xavier's College (Autonomous), Ahmedabad", "St. Xavier's College, Ahmedabad",
                "St. Xavier's College", "Xavier's Assistant", "St. Xavier's", "SXCA"]
_MARKER = re.compile(r"\[\[\s*(\d+)\s*\]\]")


def lock_names(text: str) -> tuple[str, dict[str, str]]:
    from app.config import get_settings

    names = sorted(set(LOCKED_NAMES + [get_settings().app_name]), key=len, reverse=True)
    found: dict[str, str] = {}
    for name in names:
        if name.lower() in text.lower():
            key = str(len(found) + 1)
            text = re.sub(re.escape(name), f"[[{key}]]", text, flags=re.I)
            found[key] = name
    return text, found


def unlock_names(text: str, found: dict[str, str]) -> str:
    """Put the names back; TranslationError if the model dropped a marker."""
    missing = [k for k in found if not re.search(rf"\[\[\s*{k}\s*\]\]", text)]
    if missing:
        raise TranslationError("translation dropped names: " + ", ".join(found[k] for k in missing))
    return _MARKER.sub(lambda m: found.get(m.group(1), m.group(0)), text)


async def _llm_translate(text: str, src: str, tgt: str) -> str:
    from app.llm.base import ChatMessage
    from app.llm.factory import get_llm

    found: dict[str, str] = {}
    if src == "en":
        text, found = lock_names(text)
    prompt = LLM_PROMPT.format(src=LANGUAGES[src].name, tgt=LANGUAGES[tgt].name)
    if found:
        prompt += " Copy markers like [[1]] unchanged: they stand for names."
    result = await get_llm().generate(prompt, [ChatMessage(role="user", content=text)])
    return unlock_names((result.text or "").strip(), found)


# ---------------------------------------------------------------- public API
_cache: OrderedDict[tuple[str, str, str], str] = OrderedDict()
_CACHE_MAX = 512


def providers() -> list[str]:
    """Translators to try, in order. `auto`: the LLM first (it keeps names, course codes and dates
    intact; the distilled IndicTrans2 models jumble names), IndicTrans2 as the backup when the LLM's
    output fails the checks (e.g. it returned the text untranslated)."""
    p = get_settings().translation_provider.lower()
    if p == "auto":
        return ["llm", "indictrans2"] if IndicTrans2.available() else ["llm"]
    return [] if p == "off" else [p]


def provider() -> str:
    """Main translator, for display: llm, indictrans2 or off."""
    chain = providers()
    return chain[0] if chain else "off"


def _problems(source: str, result: str, src: str, tgt: str) -> list[str]:
    """Why a translation can't be used (empty list = fine)."""
    from app.i18n.languages import detect_language

    if not result.strip():
        return ["empty"]
    problems = []
    if detect_language(result, preferred=tgt) != tgt and _LETTERS.search(source):
        problems.append(f"not in {LANGUAGES[tgt].name}")
    if len(result) > 3 * len(source) + 60:
        problems.append("much longer than the original (answered instead of translating?)")
    if src == "en":
        problems += [f"missing {m}" for m in check_translation(source, result)]
    return problems


async def _run(p: str, text: str, src: str, tgt: str) -> str:
    if p == "indictrans2":
        lines = _split(text)
        flat = [s for _, sents in lines for s in sents]
        done = iter(await asyncio.to_thread(_indictrans.translate_sentences, flat, src, tgt)) if flat else iter(())
        return _join([(prefix, [next(done) for _ in sents]) for prefix, sents in lines])
    return await _llm_translate(text, src, tgt)


async def translate(text: str, src: str, tgt: str) -> str:
    """Translate markdown-ish chat text. Raises TranslationError if no translator produced a safe result."""
    if src == tgt or not text.strip():
        return text
    if src not in LANGUAGES or tgt not in LANGUAGES:
        raise TranslationError(f"unsupported language pair {src}->{tgt}")
    key = (src, tgt, text)
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]

    chain = providers()
    if not chain:
        raise TranslationError("translation is turned off")
    failures = []
    for p in chain:
        try:
            result = (await _run(p, text, src, tgt)).strip()
        except Exception as e:
            log.warning("Translation %s->%s with %s failed: %s", src, tgt, p, e)
            failures.append(f"{p}: {e}")
            continue
        if src == "en":
            result = repair_numbers(text, result)
        problems = _problems(text, result, src, tgt)
        if not problems:
            if len(text) <= 1000:
                _cache[key] = result
                if len(_cache) > _CACHE_MAX:
                    _cache.popitem(last=False)
            return result
        log.warning("Translation %s->%s with %s rejected: %s", src, tgt, p, problems[:5])
        failures.append(f"{p}: {', '.join(problems[:3])}")
    raise TranslationError("; ".join(failures))


def warmup() -> None:
    """Load the IndicTrans2 models in the background when they are downloaded."""
    if "indictrans2" in providers():
        _indictrans.warmup()
