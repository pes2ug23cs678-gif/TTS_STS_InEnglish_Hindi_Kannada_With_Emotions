import ctranslate2.converters.transformers as _ct2m
_orig = _ct2m.TransformersConverter.load_model

def _fixed(self, *args, **kwargs):
    if "dtype" in kwargs:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
    return _orig(self, *args, **kwargs)

_ct2m.TransformersConverter.load_model = _fixed

import os
from ctranslate2.converters.transformers import TransformersConverter

out = os.path.expanduser("~/Python312_Project/ct2_models/whisper-kannada-medium")
TransformersConverter("vasista22/whisper-kannada-medium").convert(
    out, quantization="float16", force=True
)
print("Done:", out)
