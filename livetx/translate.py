"""Armenian text -> English text with NLLB-200 (CTranslate2, int8 on CPU).

The model must be converted once (see README):
    ct2-transformers-converter --model facebook/nllb-200-distilled-600M \
        --output_dir models/nllb-200-distilled-600M-ct2 --quantization int8
"""
from __future__ import annotations

from pathlib import Path

import ctranslate2
from transformers import AutoTokenizer

HF_MODEL = "facebook/nllb-200-distilled-600M"
SRC, TGT = "hye_Armn", "eng_Latn"


class Translator:
    def __init__(self, model_dir: Path, threads: int = 4):
        if not (model_dir / "model.bin").exists():
            raise FileNotFoundError(
                f"No CTranslate2 model in {model_dir}. Convert it first:\n"
                f"  ct2-transformers-converter --model {HF_MODEL} --output_dir {model_dir} --quantization int8"
            )
        self.translator = ctranslate2.Translator(str(model_dir), device="cpu", intra_threads=threads)
        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, src_lang=SRC)
        self.translate("Բարև")  # warm-up, same reason as in asr.py

    def translate(self, text: str) -> str:
        tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
        # Cap the English length: a garbled input can make NLLB run on to its 256-token limit
        # (one line took 9 s on Oct 31). English needs fewer tokens than the Armenian it comes from, so 1.5x plus a
        # margin should not cut a real line.
        result = self.translator.translate_batch([tokens], target_prefix=[[TGT]], beam_size=2,
                                                 max_decoding_length=int(1.5 * len(tokens)) + 10)
        out = result[0].hypotheses[0][1:]  # drop the target-language token
        return self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(out), skip_special_tokens=True)
