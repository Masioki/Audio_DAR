from model.utils import _load_model

BACKBONES = {
    "Phi-3-mini-4k-instruct": (3072, lambda **kwargs: _load_model('microsoft/Phi-3-mini-4k-instruct', **kwargs)),
    "distilbert-base-uncased": (768, lambda **kwargs: _load_model('distilbert/distilbert-base-uncased', **kwargs)),
    'whisper-encoder-tiny': (3072, lambda **kwargs: _load_model("openai/whisper-tiny.en", **kwargs).encoder),
}
