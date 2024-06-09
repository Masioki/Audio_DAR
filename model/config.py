from peft import LoraConfig, get_peft_model

from model.layers.prosody_encoder import ProsodyEncoder, TransformerProsodyEncoder
from model.utils import _load_model

BACKBONES = {
    "Phi-3-mini-4k-instruct": (3072, lambda **kwargs: _load_model('microsoft/Phi-3-mini-4k-instruct', **kwargs)),
    "Phi-3-mini-4k-instruct-lora": (
    3072, lambda **kwargs: _to_lora(_load_model('microsoft/Phi-3-mini-4k-instruct', **kwargs))),
    "distilbert-base-uncased": (768, lambda **kwargs: _load_model('distilbert/distilbert-base-uncased', **kwargs)),
    "distilbert-base-uncased-lora": (
    768, lambda **kwargs: _to_lora(_load_model('distilbert/distilbert-base-uncased', **kwargs))),
    "distilbert-base-cased": (768, lambda **kwargs: _load_model('distilbert/distilbert-base-cased', **kwargs)),
    'whisper-encoder-tiny': (384, lambda **kwargs: _load_model("openai/whisper-tiny.en", **kwargs).encoder),
    'whisper-encoder-small': (768, lambda **kwargs: _load_model("openai/whisper-small.en", **kwargs).encoder),
    'mpnet-base-v2': (768, lambda **kwargs: _load_model("sentence-transformers/all-mpnet-base-v2", **kwargs)),
    "lstm-prosody-encoder192": (192, lambda **kwargs: ProsodyEncoder(hidden_size=192, **kwargs)),
    "transformer-prosody-encoder192": (192, lambda **kwargs: TransformerProsodyEncoder(hidden_size=192, **kwargs)),
}

_DEFAULT_LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_lin", "k_lin", "v_lin"],
    bias='none',
)


def _to_lora(model, config=_DEFAULT_LORA_CONFIG):
    return get_peft_model(model, config)
