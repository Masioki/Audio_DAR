import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

from model.config import BACKBONES
from model.layers.cls_head import SentenceClassifierHead
from model.layers.cross_attention import CrossAttention
from model.layers.lambd import LambdaLayer
from model.layers.pooling import ConfigurablePooling
from model.utils import freeze


class CrossAttentionSentenceClassifierConfig(PretrainedConfig):
    model_type = 'cross-attention-sentence-classifier'

    def __init__(self,
                 q_backbone: str = "Phi-3-mini-4k-instruct",
                 q_freezed: bool = True,
                 q_kwargs: dict = {},
                 k_backbone: str = "Phi-3-mini-4k-instruct",
                 k_freezed: bool = True,
                 k_kwargs: dict = {},
                 heads: int = 8,
                 labels: int = 18,
                 multilabel=False,
                 embedding_strategy: str = 'wmean-pooling',
                 hidden_size: int = 768,
                 dropout: float = 0.3,
                 **kwargs):
        super().__init__(**kwargs)
        self.q_backbone = q_backbone
        self.q_kwargs = q_kwargs
        self.q_freezed = q_freezed
        self.k_backbone = k_backbone
        self.k_kwargs = k_kwargs
        self.k_freezed = k_freezed
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.embedding_strategy = embedding_strategy
        self.labels = labels
        self.multilabel = multilabel
        self.heads = heads


class CrossAttentionSentenceClassifier(PreTrainedModel):
    config_class = CrossAttentionSentenceClassifierConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        q_features, self.q_backbone = BACKBONES[config.q_backbone]
        self.q_backbone_initialized = False
        self.q_hidden_state_extractor = LambdaLayer(
            lambda input_ids, attention_mask=None, **kwargs:
            self._get_q_backbone()(input_ids, attention_mask=attention_mask, output_hidden_states=True, **kwargs)
            .hidden_states[-1]
        )

        k_features, self.k_backbone = BACKBONES[config.k_backbone]
        self.k_backbone_initialized = False
        self.k_hidden_state_extractor = LambdaLayer(
            lambda input_ids, attention_mask=None, **kwargs:
            self._get_k_backbone()(input_ids, attention_mask=attention_mask, output_hidden_states=True, **kwargs)
            .hidden_states[-1]
        )

        self.cross_attention = CrossAttention(
            q_features,
            k_features,
            config.heads,
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

    def _get_k_backbone(self):
        if self.k_backbone_initialized:
            return self.k_backbone
        self.k_backbone_initialized = True
        self.k_backbone = self.k_backbone(**self.config.k_kwargs)
        self.k_backbone.to(self.device)
        if self.config.k_freezed:
            self.k_backbone = freeze(self.k_backbone)
        return self.k_backbone

    def forward(self, q_inputs, k_inputs, q_attention_mask=None, q_hidden_states=None, k_attention_mask=None,
                k_hidden_states=None, labels=None, **kwargs):
        if q_hidden_states is None:
            q_outputs = self.q_hidden_state_extractor(q_inputs, q_attention_mask, **kwargs)
        else:
            q_outputs = q_hidden_states

        if k_hidden_states is None:
            k_outputs = self.k_hidden_state_extractor(k_inputs, k_attention_mask, **kwargs)
        else:
            k_outputs = k_hidden_states

        outputs = self.cross_attention(q_outputs, k_outputs, mask=k_attention_mask)
        outputs = self.layer_normalization(outputs)
        outputs = self.pooling(outputs, q_inputs, q_attention_mask, **kwargs)
        logits = self.head(outputs)
        loss = None
        if labels is not None:
            loss = self.loss(logits, labels.float())
        return {"logits": logits, "loss": loss}
