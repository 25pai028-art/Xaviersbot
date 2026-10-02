"""Download the IndicTrans2 translation models once and check that they translate correctly.

    hf auth login            # paste a Hugging Face read token (after accepting the models' terms)
    python -m scripts.download_translation
"""
from __future__ import annotations

import asyncio
import sys
import time

from app.i18n import translate as tr

SAMPLE = "The last date to pay the semester fees is 25 June 2026. The BCA fee is Rs. 62,500 per year."


async def main() -> int:
    t = time.time()
    try:
        await asyncio.to_thread(tr._indictrans.warmup)
    except Exception as e:  # gated repo, no token, no network
        print(f"Could not download the models: {e}\n")
        print("Accept the terms on both model pages, then run `hf auth login` with a read token (see README).")
        return 1
    print(f"Models ready in {time.time() - t:.0f}s\n")
    for lang in ("hi", "gu", "ml", "ta"):
        t = time.time()
        sentences = await asyncio.to_thread(tr._indictrans.translate_sentences, [SAMPLE], "en", lang)
        missing = tr.check_translation(SAMPLE, sentences[0])
        status = "OK" if not missing else f"MISSING {missing}"
        print(f"{lang} ({time.time() - t:.1f}s, {status}): {sentences[0]}")
    back = await asyncio.to_thread(tr._indictrans.translate_sentences, ["ബിസിഎ കോഴ്സിന്റെ ഫീസ് എത്രയാണ്?"], "ml", "en")
    print(f"ml -> en: {back[0]}")
    print("\nDone. Restart the server; translation now uses IndicTrans2.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
