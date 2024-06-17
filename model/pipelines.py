import requests
import torch
from transformers import Pipeline
from transformers.pipelines.audio_classification import ffmpeg_read

from model.config import STT_BACKBONES
from model.cross_attention_dac import CrossAttentionSentenceClassifier
from model.fusion_attention_dac import FusionCrossAttentionSentenceClassifier
from model.single_embedding_dac import SingleEmbeddingSentenceClassifier
from model.utils import combine_sequences
from utils.conversation import Conversation


class _ConfigurablePipeline(Pipeline):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        post_kwargs = {}
        if "text_tokenizer" in kwargs:
            preprocess_kwargs["text_tokenizer"] = kwargs["text_tokenizer"]
        if "audio_processor" in kwargs:
            preprocess_kwargs["audio_processor"] = kwargs["audio_processor"]
        if "sampling_rate" in kwargs:
            preprocess_kwargs["sampling_rate"] = kwargs["sampling_rate"]
        if "audio_features" in kwargs:
            preprocess_kwargs["audio_features"] = kwargs["audio_features"]
        if "id_2_label" in kwargs:
            post_kwargs["id_2_label"] = kwargs["id_2_label"]
        return preprocess_kwargs, {}, post_kwargs

    def _extract_prosody(self, audio, features, sampling_rate):
        c = Conversation(['default'], ['default'], [audio])
        seqs = {str(i): [f(c, sampling_rate, sampling_rate).get(0)] for i, f in enumerate(features)}
        return torch.stack(combine_sequences(seqs, "prosody", list(seqs.keys()))["prosody"]).float()

    def get_stt_model(self, config, fallback_model):
        if type(self.model) == SingleEmbeddingSentenceClassifier:
            return STT_BACKBONES[fallback_model]()
        if type(self.model) == CrossAttentionSentenceClassifier:
            if config.k_backbone not in STT_BACKBONES:
                return STT_BACKBONES[fallback_model]()
            return STT_BACKBONES[config.k_backbone]()
        if type(self.model) == FusionCrossAttentionSentenceClassifier:
            if config.k2_backbone not in STT_BACKBONES:
                return STT_BACKBONES[fallback_model]()
            return STT_BACKBONES[config.k1_backbone]()

    def get_name_map(self):
        return {
            "q_inputs": "q_inputs",
            "q_attention_mask": "q_attention_mask",
            "k1_inputs": "k1_inputs",
            "k2_inputs": "k2_inputs"
        }

    def preprocess(self, inputs, **arguments):
        if isinstance(inputs, str):
            if inputs.startswith("http://") or inputs.startswith("https://"):
                # We need to actually check for a real protocol, otherwise it's impossible to use a local file
                # like http_huggingface_co.png
                inputs = requests.get(inputs).content
            else:
                with open(inputs, "rb") as f:
                    inputs = f.read()

        if isinstance(inputs, bytes):
            inputs = ffmpeg_read(inputs, arguments['sampling_rate'])

        audio_inputs_features = arguments['audio_processor']([inputs], sampling_rate=arguments['sampling_rate'],
                                                             return_tensors="pt").input_features
        predicted_ids = self.get_stt_model(self.model.config, list(STT_BACKBONES.keys())[0]).generate(
            audio_inputs_features)
        transcription = arguments['audio_processor'].batch_decode(predicted_ids, skip_special_tokens=True)[0]
        prosody_features = None
        if "audio_features" in arguments:
            prosody_features = self._extract_prosody(inputs, arguments['audio_features'],
                                                     arguments['sampling_rate'])
        text_inputs = arguments['text_tokenizer'](transcription, truncation=True, padding=True, max_length=512,
                                                  return_tensors="pt")

        model_input = {
            "q_inputs": text_inputs["input_ids"],
            "q_attention_mask": text_inputs["attention_mask"],
            "k1_inputs": audio_inputs_features,
            "k2_inputs": prosody_features,
        }

        model_input = {v: model_input[k] for k, v in self.get_name_map().items()}

        return {"model_inputs": model_input, "transcript": transcription}

    def _forward(self, model_inputs, **kwargs):
        outputs = self.model(**model_inputs["model_inputs"])
        return outputs, model_inputs["transcript"]

    def postprocess(self, model_outputs, **kwargs):
        best_class = torch.sigmoid(model_outputs[0]["logits"])[0] > 0.5
        if "id_2_label" in kwargs:
            best_class = [kwargs["id_2_label"][i] for i, val in enumerate(best_class.long()) if val == 1]
        return best_class, model_outputs[1]


class SingleEmbeddingPipeline(_ConfigurablePipeline):

    def get_name_map(self):
        return {
            "q_inputs": "input_ids",
            "q_attention_mask": "attention_mask"
        }


class CrossAttentionPipeline(_ConfigurablePipeline):

    def get_name_map(self):
        if "prosody" in self.model.config.k_backbone:
            return {
                "q_inputs": "q_inputs",
                "q_attention_mask": "q_attention_mask",
                "k2_inputs": "k_inputs"
            }
        return {
            "q_inputs": "q_inputs",
            "q_attention_mask": "q_attention_mask",
            "k1_inputs": "k_inputs"
        }


class FusionPipeline(_ConfigurablePipeline):
    pass
