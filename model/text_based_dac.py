import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

from model.config import BACKBONES
from model.utils import wmean_pooling, mean_pooling, cls_token, freeze


class TextBasedSentenceClassifierConfig(PretrainedConfig):
    model_type = 'text-based-sentence-classifier'

    def __init__(self, backbone: str = "Phi-3-mini-4k-instruct", labels: int = 18, multilabel=False,
                 embedding_strategy: str = 'wmean-pooling',
                 backbone_freezed: bool = True,
                 backbone_kwargs: dict = {}, hidden_size: int = 768, dropout: float = 0.1,
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


class TextBasedSentenceClassifier(PreTrainedModel):
    config_class = TextBasedSentenceClassifierConfig

    def __init__(self, config):
        super().__init__(config)
        features, _ = BACKBONES[config.backbone]
        self.config = config
        if config.multilabel:
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = nn.CrossEntropyLoss()
        self.sentence_embedding = LlmBasedEmbedding(config)
        if config.backbone_freezed:
            self.sentence_embedding = freeze(self.sentence_embedding.backbone)
        self.head = SentenceClassifierHead(features, config)

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        sentence_embeddings = self.sentence_embedding(input_ids, attention_mask, **kwargs)
        logits = self.head(sentence_embeddings)
        loss = None
        if labels is not None:
            loss = self.loss(logits, labels.float())
        return {"logits": logits, "loss": loss}


class SentenceClassifierHead(nn.Module):
    def __init__(self, features, config):
        super().__init__()
        self.config = config
        self.model = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(features, self.config.hidden_size),
            nn.Tanh(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_size, self.config.labels),
        )

    def forward(self, x):
        return self.model(x)


class LlmBasedEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = BACKBONES[config.backbone][1](**config.backbone_kwargs)
        self.flatten = nn.Flatten(start_dim=1)
        self.hidden_state_extractor = lambda input_ids, attention_mask=None, **kwargs: \
            self.backbone(input_ids, attention_mask=attention_mask, output_hidden_states=True, **kwargs).hidden_states[
                -1]
        if config.embedding_strategy == 'wmean-pooling':
            self.sentence_embedding = wmean_pooling
        elif config.embedding_strategy == 'mean-pooling':
            self.sentence_embedding = mean_pooling
        elif config.embedding_strategy == 'cls':
            self.sentence_embedding = cls_token
        elif config.embedding_strategy == 'provided':
            self.hidden_state_extractor = lambda input_ids, attention_mask=None, **kwargs: self.backbone(input_ids,
                                                                                                         attention_mask=attention_mask,
                                                                                                         **kwargs)
            self.sentence_embedding = lambda outputs, input_ids, attention_mask, **kwargs: outputs
        else:
            self.sentence_embedding = lambda outputs, input_ids, attention_mask, **kwargs: self.flatten(outputs)

    def get_embedding_before_pooling(self, input_ids, attention_mask=None, **kwargs):
        return self.hidden_state_extractor(input_ids, attention_mask, **kwargs)

    def forward(self, input_ids, attention_mask=None, hidden_states=None, **kwargs):
        if hidden_states is None:
            output = self.get_embedding_before_pooling(input_ids, attention_mask, **kwargs)
        else:
            output = hidden_states.to(self.device)
        return self.sentence_embedding(output, input_ids, attention_mask, **kwargs)
