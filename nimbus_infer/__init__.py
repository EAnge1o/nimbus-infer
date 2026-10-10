"""nimbus-infer: a from-scratch, NumPy-only LLM inference engine."""

from .backend import DTYPE
from .cache import KVCache
from .config import ModelConfig, get_preset
from .engine import Engine
from .model import Transformer, init_params
from .quant import dequantize, quantize
from .sampling import SamplingParams, sample
from .tokenizer import BPETokenizer
from .weights import read_safetensors, save_safetensors

__version__ = "0.1.0"

__all__ = [
    "BPETokenizer",
    "DTYPE",
    "Engine",
    "KVCache",
    "ModelConfig",
    "SamplingParams",
    "Transformer",
    "dequantize",
    "get_preset",
    "init_params",
    "quantize",
    "read_safetensors",
    "save_safetensors",
    "sample",
]