import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

from model.config import BACKBONES
from model.utils import wmean_pooling, mean_pooling, cls_token


class TextBasedSentenceClassifierConfig(PretrainedConfig):
    model_type = 'text-based-sentence-classifier'

    def __init__(self, backbone: str, labels: int, multilabel=False, embedding_strategy: str = 'wmean-pooling',
                 backbone_kwargs: dict = {}, backbone_module: str = None, hidden_size: int = 768, dropout: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.backbone_kwargs = backbone_kwargs
        self.backbone_module = backbone_module
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.embedding_strategy = embedding_strategy
        self.labels = labels
        self.multilabel = multilabel


class TextBasedSentenceClassifier(PreTrainedModel):
    config_class = TextBasedSentenceClassifierConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        features, self.backbone = BACKBONES[config.backbone]
        self.backbone = self.backbone(**config.backbone_kwargs)

        if config.multilabel:
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = nn.CrossEntropyLoss()
        self.flatten = nn.Flatten(start_dim=1)

        if config.embedding_strategy == 'wmean-pooling':
            self.sentence_embedding = wmean_pooling
        elif config.embedding_strategy == 'mean-pooling':
            self.sentence_embedding = mean_pooling
        elif config.embedding_strategy == 'cls':
            self.sentence_embedding = cls_token
        elif config.embedding_strategy == 'provided':
            self.sentence_embedding = lambda model, input_ids, attention_mask, **kwargs: model(input_ids,
                                                                                               attention_mask=attention_mask,
                                                                                               **kwargs)
        else:
            self.sentence_embedding = lambda model, input_ids, attention_mask, **kwargs: self.flatten(
                model(input_ids, attention_mask=attention_mask, output_hidden_states=True, **kwargs).hidden_states[-1])

        self.model = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(features, self.config.hidden_size),
            nn.Tanh(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_size, self.config.labels),
        )

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        sentence_embeddings = self.sentence_embedding(self.backbone, input_ids, attention_mask, **kwargs)
        logits = self.model(sentence_embeddings)
        loss = None
        if labels is not None:
            loss = self.loss(logits, labels.float())
        return {"logits": logits, "loss": loss}
