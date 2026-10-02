"""Translation between English and Indian languages.

Providers (`TRANSLATION_PROVIDER`):
- `indictrans2`: AI4Bharat IndicTrans2 distilled models, run locally (free, private, works offline).
- `llm`: the configured answering LLM translates (good with Gemini/Claude; weak with small local models).
- `auto` (default): IndicTrans2 if its models are downloaded, otherwise the LLM.

Safety: the answer is fact-checked in English first. After translation every number, email and URL
of the English text must still be present; otherwise the translation is rejected (TranslationError)
and the caller shows the English answer instead of a translation that might have changed a fee or date.
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import re
import sys
import threading
import types
from collections import OrderedDict

from app.config import get_settings
from app.i18n.languages import LANGUAGES

log = logging.getLogger(__name__)

EN_INDIC = "ai4bharat/indictrans2-en-indic-dist-200M"
INDIC_EN = "ai4bharat/indictrans2-indic-en-dist-200M"
BATCH = 16
MAX_SENTENCE_CHARS = 600

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"https?://[^\s<>\"')\]]+")
_NUMBER = re.compile(r"\d+(?:[.,:/-]\d+)*")
_BULLET = re.compile(r"^(\s*(?:[-*•+]|\d+[.)])\s+)(.*)$")
_SENTENCE_END = re.compile(r"(?<=[.!?।؟])\s+(?=\S)")
# A full stop after these is not the end of a sentence ("Rs. 62,500", "Dr. Pravida", "St. Xavier's").
_ABBREVIATION = re.compile(
    r"(?<![\w'’])(?:rs|dr|mr|mrs|ms|prof|st|sr|jr|no|nos|vs|etc|e\.g|i\.e|approx|dept|govt|sem|ph|b|m)\.$", re.I)
_INITIAL = re.compile(r"(?<![\w'’])[A-Z]\.$")  # "A. C." in a name


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    for part in _SENTENCE_END.split(text):
        if out and (_ABBREVIATION.search(out[-1]) or _INITIAL.search(out[-1])):
            out[-1] += " " + part
        else:
            out.append(part)
    return [s for s in out if s.strip()]
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
        sentences = _sentences(body)
        # Very long "sentences" (tables flattened into one line) are cut so the model sees all of them.
        pieces = []
        for s in sentences:
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


# ---------------------------------------------------------------- IndicTrans2
def _indic_processor():
    """IndicTransToolkit's IndicProcessor (script unification, placeholders for numbers/URLs/emails).
    Imported without the package __init__, whose training helpers don't load with transformers 5."""
    if "IndicTransToolkit.processor" not in sys.modules:
        spec = importlib.util.find_spec("IndicTransToolkit")
        if spec is None or not spec.submodule_search_locations:
            raise TranslationError("IndicTransToolkit is not installed")
        pkg = types.ModuleType("IndicTransToolkit")
        pkg.__path__ = list(spec.submodule_search_locations)
        sys.modules.setdefault("IndicTransToolkit", pkg)
    from IndicTransToolkit.processor import IndicProcessor

    return IndicProcessor(inference=True)


class IndicTrans2:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: dict[str, tuple] = {}
        self._ip = None

    @staticmethod
    def available() -> bool:
        """Both models are downloaded (checked without network access)."""
        try:
            from huggingface_hub import try_to_load_from_cache

            return all(isinstance(try_to_load_from_cache(m, "config.json"), str) for m in (EN_INDIC, INDIC_EN))
        except Exception:
            return False

    def _model(self, name: str):
        if name not in self._models:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            kw = {"trust_remote_code": True}
            try:
                tok = AutoTokenizer.from_pretrained(name, local_files_only=True, **kw)
                model = AutoModelForSeq2SeqLM.from_pretrained(name, local_files_only=True, **kw)
            except OSError:
                tok = AutoTokenizer.from_pretrained(name, **kw)
                model = AutoModelForSeq2SeqLM.from_pretrained(name, **kw)
            torch.set_num_threads(max(1, torch.get_num_threads()))
            self._models[name] = (tok, model.eval())
            log.info("Loaded translation model %s", name)
        return self._models[name]

    def warmup(self) -> None:
        with self._lock:
            self._ip = self._ip or _indic_processor()
            self._model(EN_INDIC)
            self._model(INDIC_EN)

    def translate_sentences(self, sentences: list[str], src: str, tgt: str) -> list[str]:
        import torch

        s_tag, t_tag = LANGUAGES[src].flores, LANGUAGES[tgt].flores
        with self._lock:
            self._ip = self._ip or _indic_processor()
            tok, model = self._model(EN_INDIC if src == "en" else INDIC_EN)
            out: list[str] = []
            for i in range(0, len(sentences), BATCH):
                batch = self._ip.preprocess_batch(sentences[i:i + BATCH], src_lang=s_tag, tgt_lang=t_tag)
                enc = tok(batch, truncation=True, padding="longest", max_length=256, return_tensors="pt")
                with torch.inference_mode():
                    gen = model.generate(**enc, num_beams=get_settings().translation_beams, max_length=256,
                                         use_cache=True)
                decoded = tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                out += self._ip.postprocess_batch(decoded, lang=t_tag)
        return out


_indictrans = IndicTrans2()


# ---------------------------------------------------------------- LLM fallback
LLM_PROMPT = (
    "Translate the user's text from {src} to {tgt}. Keep every number, date, amount, phone number, email "
    "address, URL, person's name and course code exactly as written. Keep line breaks and bullet markers. "
    "Output ONLY the translation."
)


async def _llm_translate(text: str, src: str, tgt: str) -> str:
    from app.llm.base import ChatMessage
    from app.llm.factory import get_llm

    prompt = LLM_PROMPT.format(src=LANGUAGES[src].name, tgt=LANGUAGES[tgt].name)
    result = await get_llm().generate(prompt, [ChatMessage(role="user", content=text)])
    return (result.text or "").strip()


# ---------------------------------------------------------------- public API
_cache: OrderedDict[tuple[str, str, str], str] = OrderedDict()
_CACHE_MAX = 512


def provider() -> str:
    """The provider that will actually be used: indictrans2, llm or off."""
    p = get_settings().translation_provider.lower()
    if p == "auto":
        return "indictrans2" if IndicTrans2.available() else "llm"
    return p


async def translate(text: str, src: str, tgt: str) -> str:
    """Translate markdown-ish chat text. Raises TranslationError if it can't be done safely."""
    if src == tgt or not text.strip():
        return text
    if src not in LANGUAGES or tgt not in LANGUAGES:
        raise TranslationError(f"unsupported language pair {src}->{tgt}")
    key = (src, tgt, text)
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]

    p = provider()
    if p == "off":
        raise TranslationError("translation is turned off")
    try:
        if p == "indictrans2":
            lines = _split(text)
            flat = [s for _, sents in lines for s in sents]
            done = iter(await asyncio.to_thread(_indictrans.translate_sentences, flat, src, tgt)) if flat else iter(())
            result = _join([(prefix, [next(done) for _ in sents]) for prefix, sents in lines])
        else:
            result = await _llm_translate(text, src, tgt)
    except TranslationError:
        raise
    except Exception as e:
        log.warning("Translation %s->%s failed: %s", src, tgt, e)
        raise TranslationError(str(e)) from e

    if not result:
        raise TranslationError("empty translation")
    missing = check_translation(text, result) if src == "en" else []
    if missing:
        log.warning("Translation %s->%s dropped %s; using English", src, tgt, missing[:5])
        raise TranslationError("translation changed protected facts: " + ", ".join(missing[:5]))

    if len(text) <= 1000:
        _cache[key] = result
        if len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return result


def warmup() -> None:
    """Load the IndicTrans2 models in the background when they are downloaded."""
    if provider() == "indictrans2":
        _indictrans.warmup()
