"""Tokenizer for IndicTrans2, rewritten without `transformers.PreTrainedTokenizer` (whose internals changed in
transformers 5). It reproduces what the original does for inference: SentencePiece pieces looked up in the
model's vocabulary files, source/target language tags first, `</s>` at the end, padding on the left.
Based on AI4Bharat's tokenization_indictrans.py (MIT licence)."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from sentencepiece import SentencePieceProcessor

LANGUAGE_TAGS = frozenset({
    "asm_Beng", "awa_Deva", "ben_Beng", "bho_Deva", "brx_Deva", "doi_Deva", "eng_Latn", "gom_Deva",
    "gon_Deva", "guj_Gujr", "hin_Deva", "hne_Deva", "kan_Knda", "kas_Arab", "kas_Deva", "kha_Latn",
    "lus_Latn", "mag_Deva", "mai_Deva", "mal_Mlym", "mar_Deva", "mni_Beng", "mni_Mtei", "npi_Deva",
    "ory_Orya", "pan_Guru", "san_Deva", "sat_Olck", "snd_Arab", "snd_Deva", "tam_Taml", "tel_Telu",
    "urd_Arab", "unr_Deva",
})


class IndicTransTokenizer:
    def __init__(self, model_dir: str | Path):
        d = Path(model_dir)
        self.src_encoder: dict[str, int] = json.loads((d / "dict.SRC.json").read_text(encoding="utf-8"))
        self.tgt_encoder: dict[str, int] = json.loads((d / "dict.TGT.json").read_text(encoding="utf-8"))
        self.tgt_decoder = {v: k for k, v in self.tgt_encoder.items()}
        self.src_spm = SentencePieceProcessor(model_file=str(d / "model.SRC"))
        self.tgt_spm = SentencePieceProcessor(model_file=str(d / "model.TGT"))
        self.unk_id = self.src_encoder["<unk>"]
        self.pad_id = self.src_encoder["<pad>"]
        self.eos_id = self.src_encoder["</s>"]
        self.bos_id = self.src_encoder["<s>"]
        self._special_tgt = {self.tgt_encoder.get(t) for t in ("<unk>", "<pad>", "</s>", "<s>")}

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "IndicTransTokenizer":
        return cls(model_dir)

    def _encode(self, text: str, max_length: int) -> list[int]:
        src_lang, tgt_lang, body = text.split(" ", 2)
        if src_lang not in LANGUAGE_TAGS or tgt_lang not in LANGUAGE_TAGS:
            raise ValueError(f"invalid language tags: {src_lang} {tgt_lang}")
        pieces = [src_lang, tgt_lang] + self.src_spm.EncodeAsPieces(body)
        ids = [self.src_encoder.get(p, self.unk_id) for p in pieces][: max_length - 1]
        return ids + [self.eos_id]

    def __call__(self, batch: list[str], max_length: int = 256, **_ignored) -> dict[str, torch.Tensor]:
        """Pre-processed sentences ("eng_Latn mal_Mlym text") → left-padded input_ids and attention_mask."""
        encoded = [self._encode(t, max_length) for t in batch]
        width = max(len(e) for e in encoded)
        ids = [[self.pad_id] * (width - len(e)) + e for e in encoded]
        mask = [[0] * (width - len(e)) + [1] * len(e) for e in encoded]
        return {"input_ids": torch.tensor(ids, dtype=torch.long), "attention_mask": torch.tensor(mask, dtype=torch.long)}

    def batch_decode(self, sequences, skip_special_tokens: bool = True, **_ignored) -> list[str]:
        out = []
        for seq in sequences:
            ids = seq.tolist() if hasattr(seq, "tolist") else list(seq)
            if skip_special_tokens:
                ids = [i for i in ids if i not in self._special_tgt]
            tokens = [self.tgt_decoder.get(i, "<unk>") for i in ids]
            out.append("".join(tokens).replace("▁", " ").strip())
        return out
