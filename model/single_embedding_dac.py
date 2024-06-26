import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

from model.config import BACKBONES
from model.layers.cls_head import SentenceClassifierHead
from model.layers.lambd import LambdaLayer
from model.layers.pooling import ConfigurablePooling
from model.utils import freeze


class SingleEmbeddingSentenceClassifierConfig(PretrainedConfig):
    model_type = 'single-embedding-sentence-classifier'

    def __init__(self, backbone: str = "Phi-3-mini-4k-instruct", labels: int = 18, multilabel=False,
                 embedding_strategy: str = 'wmean-pooling',
                 backbone_freezed: bool = True,
                 backbone_kwargs: dict = {}, hidden_size: int = 768, dropout: float = 0.3,
                 **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.backbone_kwargs = backbone_kwargs
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.embedding_strategy = embedding_strategy
        self.labels = labels
        self.multilabel = multilabel
        self.backbone_freezed = backbone_freezed


class SingleEmbeddingSentenceClassifier(PreTrainedModel):
    config_class = SingleEmbeddingSentenceClassifierConfig

    def __init__(self, config, load_on_init=False):
        super().__init__(config)
        self.config = config
        features, self.backbone = BACKBONES[config.backbone]
        self.backbone_initialized = False
        if load_on_init:
            self.backbone = self._get_backbone()

        self.hidden_state_extractor = lambda input_ids, attention_mask=None, **kwargs: \
            self._get_backbone()(input_ids, attention_mask=attention_mask, output_hidden_states=True,
                                 **kwargs).hidden_states[
                -1]
        if config.embedding_strategy == 'provided':
            self.hidden_state_extractor = lambda input_ids, attention_mask=None, **kwargs: \
                self._get_backbone()(input_ids,
                                     attention_mask=attention_mask,
                                     **kwargs)[0]
        self.hidden_state_extractor = LambdaLayer(self.hidden_state_extractor)
        self.pooling = ConfigurablePooling(
            features,
            config.embedding_strategy
        )
        self.head = SentenceClassifierHead(
            features,
            config.hidden_size,
            config.labels,
            config.dropout
        )

        if self.config.multilabel:
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = nn.CrossEntropyLoss()

    def _get_backbone(self):
        if self.backbone_initialized:
            return self.backbone
        self.backbone_initialized = True
        self.backbone = self.backbone(**self.config.backbone_kwargs)
        self.backbone.to(self.device)
        if self.config.backbone_freezed:
            self.backbone = freeze(self.backbone)
        return self.backbone

    def forward(self, input_ids, labels=None, attention_mask=None, hidden_states=None, **kwargs):
        if hidden_states is None:
            outputs = self.hidden_state_extractor(input_ids, attention_mask, **kwargs)
        else:
            outputs = hidden_states
        outputs = self.pooling(outputs, input_ids, attention_mask, **kwargs)
        logits = self.head(outputs)
        loss = None
        if labels is not None:
            loss = self.loss(logits, labels.float())
        return {"logits": logits, "loss": loss}
