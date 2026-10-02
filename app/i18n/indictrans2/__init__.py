"""AI4Bharat IndicTrans2 model code (MIT licence, see LICENSE), kept here instead of loading it with
`trust_remote_code`, because the published files no longer import with transformers 5.

Changes from the files on the Hugging Face Hub (ai4bharat/indictrans2-*-dist-200M):
- configuration_indictrans.py: removed the ONNX export config (`transformers.onnx` no longer exists).
- tokenization_indictrans.py: rewritten as a small standalone tokenizer (same vocabulary files and
  behaviour for translation), since transformers 5 changed the tokenizer base class it relied on.
Running code from this package also means no code from the internet is executed at load time.
"""
from .configuration_indictrans import IndicTransConfig
from .modeling_indictrans import IndicTransForConditionalGeneration
from .tokenization_indictrans import IndicTransTokenizer

__all__ = ["IndicTransConfig", "IndicTransForConditionalGeneration", "IndicTransTokenizer"]
