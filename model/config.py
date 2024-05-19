from model.utils import freeze, _load_model

BACKBONES = {
    "phi-3-freezed": lambda **kwargs: freeze(_load_model('microsoft/Phi-3-mini-4k-instruct', **kwargs)),
    'whisper-encoder-tiny-freezed': lambda **kwargs: freeze(_load_model("openai/whisper-tiny.en", **kwargs).encoder),
}
