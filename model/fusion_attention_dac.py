import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

from model.config import BACKBONES
from model.layers.cls_head import SentenceClassifierHead
from model.layers.cross_attention_fusion import CrossAttentionFusionModule
from model.layers.lambd import LambdaLayer
from model.layers.pooling import ConfigurablePooling
from model.utils import freeze


class FusionCrossAttentionSentenceClassifierConfig(PretrainedConfig):
    model_type = 'fusion-cross-attention-sentence-classifier'

    def __init__(self,
                 q_backbone: str = "Phi-3-mini-4k-instruct",
                 q_freezed: bool = True,
                 q_kwargs: dict = {},
                 k1_backbone: str = 'whisper-encoder-small',
                 k1_freezed: bool = True,
                 k1_kwargs: dict = {},
                 k2_backbone: str = "transformer-prosody-encoder192",
                 k2_freezed: bool = True,
                 k2_kwargs: dict = {},
                 heads: int = 8,
                 labels: int = 18,
                 multilabel=False,
                 embedding_strategy: str = 'wmean-pooling',
                 fusion_strategy: str = 'dense',
                 fusion_layers: int = 2,
                 hidden_size: int = 768,
                 dropout: float = 0.3,
                 **kwargs):
        super().__init__(**kwargs)
        self.q_backbone = q_backbone
        self.q_kwargs = q_kwargs
        self.q_freezed = q_freezed
        self.k1_backbone = k1_backbone
        self.k1_kwargs = k1_kwargs
        self.k1_freezed = k1_freezed
        self.k2_backbone = k2_backbone
        self.k2_kwargs = k2_kwargs
        self.k2_freezed = k2_freezed
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.embedding_strategy = embedding_strategy
        self.labels = labels
        self.multilabel = multilabel
        self.heads = heads
        self.fusion_strategy = fusion_strategy
        self.fusion_layers = fusion_layers


class FusionCrossAttentionSentenceClassifier(PreTrainedModel):
    config_class = FusionCrossAttentionSentenceClassifierConfig

    def __init__(self, config, load_q_on_init=False, load_k1_on_init=False, load_k2_on_init=False):
        super().__init__(config)
        self.config = config
        q_features, self.q_backbone = BACKBONES[config.q_backbone]
        self.q_backbone_initialized = False
        if load_q_on_init:
            self.q_backbone = self._get_q_backbone()
        self.q_hidden_state_extractor = LambdaLayer(
            lambda input_ids, attention_mask=None, **kwargs:
            self._get_q_backbone()(input_ids, attention_mask=attention_mask, output_hidden_states=True, **kwargs)
            .hidden_states[-1]
        )

        k1_features, self.k1_backbone = BACKBONES[config.k1_backbone]
        self.k1_backbone_initialized = False
        if load_k1_on_init:
            self.k1_backbone = self._get_k1_backbone()
        self.k1_hidden_state_extractor = LambdaLayer(
            lambda input_ids, attention_mask=None, **kwargs:
            self._get_k1_backbone()(input_ids, attention_mask=attention_mask, output_hidden_states=True, **kwargs)
            .hidden_states[-1]
        )

        k2_features, self.k2_backbone = BACKBONES[config.k2_backbone]
        self.k2_backbone_initialized = False
        if load_k2_on_init:
            self.k2_backbone = self._get_k2_backbone()
        self.k2_hidden_state_extractor = LambdaLayer(
            lambda input_ids, attention_mask=None, **kwargs:
            self._get_k2_backbone()(input_ids, attention_mask=attention_mask, output_hidden_states=True, **kwargs)
            .hidden_states[-1]
        )

        self.cross_attention = CrossAttentionFusionModule(
            q_features,
            k1_features,
            k2_features,
            config.heads,
            config.fusion_strategy,
            config.fusion_layers
        )
        self.layer_normalization = nn.LayerNorm(q_features)
        self.pooling = ConfigurablePooling(
            q_features,
            config.embedding_strategy
        )
        self.head = SentenceClassifierHead(
            q_features,
            config.hidden_size,
            config.labels,
            config.dropout
        )

        if self.config.multilabel:
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = nn.CrossEntropyLoss()

    def _get_q_backbone(self):
        if self.q_backbone_initialized:
            return self.q_backbone
        self.q_backbone_initialized = True
        self.q_backbone = self.q_backbone(**self.config.q_kwargs)
        self.q_backbone.to(self.device)
        if self.config.q_freezed:
            self.q_backbone = freeze(self.q_backbone)
        return self.q_backbone

    def _get_k1_backbone(self):
        if self.k1_backbone_initialized:
            return self.k1_backbone
        self.k1_backbone_initialized = True
        self.k1_backbone = self.k1_backbone(**self.config.k1_kwargs)
        self.k1_backbone.to(self.device)
        if self.config.k1_freezed:
            self.k1_backbone = freeze(self.k1_backbone)
        return self.k1_backbone

    def _get_k2_backbone(self):
        if self.k2_backbone_initialized:
            return self.k2_backbone
        self.k2_backbone_initialized = True
        self.k2_backbone = self.k2_backbone(**self.config.k2_kwargs)
        self.k2_backbone.to(self.device)
        if self.config.k2_freezed:
            self.k2_backbone = freeze(self.k2_backbone)
        return self.k2_backbone

    def forward(self, q_inputs, k1_inputs, k2_inputs,
                q_attention_mask=None, q_hidden_states=None,
                k1_attention_mask=None, k1_hidden_states=None,
                k2_attention_mask=None, k2_hidden_states=None,
                labels=None, **kwargs):
        if q_hidden_states is None:
            q_outputs = self.q_hidden_state_extractor(q_inputs, q_attention_mask, **kwargs)
        else:
            q_outputs = q_hidden_states

        if k1_hidden_states is None:
            k1_outputs = self.k1_hidden_state_extractor(k1_inputs, k1_attention_mask, **kwargs)
        else:
            k1_outputs = k1_hidden_states

        if k2_hidden_states is None:
            k2_outputs = self.k2_hidden_state_extractor(k2_inputs, k2_attention_mask, **kwargs)
        else:
            k2_outputs = k2_hidden_states

        outputs = self.cross_attention(q_outputs, k1_outputs, k2_outputs, k1_mask=k1_attention_mask,
                                       k2_mask=k2_attention_mask)
        outputs = self.layer_normalization(outputs)
        outputs = self.pooling(outputs, q_inputs, q_attention_mask, **kwargs)
        logits = self.head(outputs)
        loss = None
        if labels is not None:
            loss = self.loss(logits, labels.float())
        return {"logits": logits, "loss": loss}
