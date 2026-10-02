"""Supported languages and script-based language detection.

The knowledge base and the answering model work in English. A question in another language is
translated to English, answered, fact-checked, and the answer is translated back.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str  # short code used by the API and the widget
    name: str  # English name (used in prompts and logs)
    native: str  # shown in the language selector
    flores: str  # IndicTrans2 language tag
    script: str | None  # regex character range of its script (None = Latin)


LANGUAGES: dict[str, Language] = {l.code: l for l in [
    Language("en", "English", "English", "eng_Latn", None),
    Language("hi", "Hindi", "हिन्दी", "hin_Deva", r"ऀ-ॿ"),
    Language("gu", "Gujarati", "ગુજરાતી", "guj_Gujr", r"઀-૿"),
    Language("ml", "Malayalam", "മലയാളം", "mal_Mlym", r"ഀ-ൿ"),
    Language("ta", "Tamil", "தமிழ்", "tam_Taml", r"஀-௿"),
    Language("te", "Telugu", "తెలుగు", "tel_Telu", r"ఀ-౿"),
    Language("kn", "Kannada", "ಕನ್ನಡ", "kan_Knda", r"ಀ-೿"),
    Language("mr", "Marathi", "मराठी", "mar_Deva", r"ऀ-ॿ"),
    Language("bn", "Bengali", "বাংলা", "ben_Beng", r"ঀ-৿"),
    Language("pa", "Punjabi", "ਪੰਜਾਬੀ", "pan_Guru", r"਀-੿"),
    Language("or", "Odia", "ଓଡ଼ିଆ", "ory_Orya", r"଀-୿"),
    Language("ur", "Urdu", "اردو", "urd_Arab", r"؀-ۿ"),
]}

# Each script maps to one language; Devanagari is Hindi unless the user picked Marathi.
_SCRIPT_LANG = {"hi": re.compile(f"[{LANGUAGES['hi'].script}]")}
for _l in LANGUAGES.values():
    if _l.script and _l.code not in ("hi", "mr"):
        _SCRIPT_LANG[_l.code] = re.compile(f"[{_l.script}]")
_LETTER = re.compile(r"[^\W\d_]")


def detect_language(text: str, preferred: str | None = None) -> str:
    """Language code from the script the text is written in. Romanised Hindi etc. counts as English
    (the answering model reads it fine). `preferred` resolves Devanagari to Marathi when chosen."""
    letters = len(_LETTER.findall(text))
    if not letters:
        return "en"
    counts = {code: len(rx.findall(text)) for code, rx in _SCRIPT_LANG.items()}
    code, n = max(counts.items(), key=lambda kv: kv[1])
    if n < max(2, letters * 0.3):
        return "en"
    if code == "hi" and preferred == "mr":
        return "mr"
    return code


def answer_language(selected: str | None, question: str) -> str:
    """The language to answer in: what the student typed in, else the language chosen in the selector.
    Typing in Malayalam gets a Malayalam answer even if the selector says English."""
    selected = selected if selected in LANGUAGES else None
    detected = detect_language(question, preferred=selected)
    if detected != "en":
        return detected
    return selected or "en"
