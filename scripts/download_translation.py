"""Download the IndicTrans2 translation models once and check that they translate correctly.

    hf auth login            # paste a Hugging Face read token (after accepting the models' terms)
    python -m scripts.download_translation
"""
from __future__ import annotations

import asyncio
import sys
import time

from app.i18n import translate as tr

ANSWERS = [
    "The last date to pay the semester fees is 25 June 2026.",
    "The B.Com fee is Rs. 62,500 per year. Email admissions@sxca.edu.in for details.",
    "Dr. Pravida Raja A.C. is the Head of the Data Science Department.",
]
QUESTIONS = [
    ("ml", "ബിസിഎ കോഴ്സിന്റെ ഫീസ് എത്രയാണ്?"), ("ml", "ബി.കോം അഡ്മിഷൻ എപ്പോഴാണ് തുടങ്ങുന്നത്?"),
    ("hi", "डेटा साइंस विभाग के प्रमुख कौन हैं?"), ("gu", "પરીક્ષાનું સમયપત્રક ક્યાં મળશે?"),
    ("ta", "எம்.காம் படிப்புக்கு தகுதி என்ன?"),
]


async def main() -> int:
    t = time.time()
    try:
        await asyncio.to_thread(tr._indictrans.warmup)
    except Exception as e:  # gated repo, no token, no network
        print(f"Could not load the models: {e}\n")
        print("Accept the terms on both model pages, then run `hf auth login` with a read token (see README).")
        return 1
    print(f"Models ready in {time.time() - t:.0f}s (translation provider: {tr.provider()})\n")
    text = "\n".join(ANSWERS)
    for lang in ("hi", "gu", "ml", "ta"):
        t = time.time()
        try:
            out = await tr.translate(text, "en", lang)
        except tr.TranslationError as e:
            out = f"[would show English: {e}]"
        print(f"--- en -> {lang} ({time.time() - t:.1f}s)\n{out}\n")
    for lang, q in QUESTIONS:
        t = time.time()
        out = await tr.translate(q, lang, "en")
        print(f"{lang} -> en ({time.time() - t:.1f}s): {out}")
    print("\nDone. Restart the server: IndicTrans2 is now the backup translator.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
