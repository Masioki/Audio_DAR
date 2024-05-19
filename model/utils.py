import os
from typing import Any

import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from transformers import AutoModel, Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback


def multi_label_compute_metrics(p):
    predictions, labels = p
    predictions = torch.sigmoid(predictions).cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    return {
        "precision": precision_score(labels, predictions > 0.5, average='micro'),
        "recall": recall_score(labels, predictions > 0.5, average='micro'),
        "f1-micro": f1_score(labels, predictions > 0.5, average='micro'),
        "f1-macro": f1_score(labels, predictions > 0.5, average='macro'),
        "accuracy": accuracy_score(labels, predictions > 0.5),
    }


def single_label_compute_metrics(p):
    predictions, labels = p
    predictions = torch.softmax(predictions, dim=-1).argmax(dim=-1).cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    return {
        "precision": precision_score(labels, predictions, average='micro'),
        "recall": recall_score(labels, predictions, average='micro'),
        "f1-micro": f1_score(labels, predictions, average='micro'),
        "f1-macro": f1_score(labels, predictions, average='macro'),
        "accuracy": accuracy_score(labels, predictions),
    }


def train(model_provider, tokenizer, root_path, name, ds, compute_metrics, epochs: int = 2, weight_decay: float = 0.01,
          tag: str = "default", patience: int = 10):
    model_output_dir = str(os.path.join(root_path, name, tag))
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=500,
        logging_strategy="steps",
        logging_steps=500,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        report_to="tensorboard"
    )

    trainer = Trainer(
        model_init=model_provider,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, padding='max_length'),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )
    trainer.train()
    trainer.push_to_hub(tags=[tag])


def _load_model(name: str, **kwargs) -> Any:
    return AutoModel.from_pretrained(name, **kwargs)


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def wmean_pooling(model, input_ids, attention_mask, **kwargs):
    last_hidden_state = \
        model(input_ids, attention_mask=attention_mask, output_hidden_states=True, **kwargs).hidden_states[-1]
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    weights_for_non_padding = attention_mask * torch.arange(start=1, end=last_hidden_state.shape[1] + 1).unsqueeze(0)
    sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
    num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
    return sum_embeddings / num_of_none_padding_tokens


def mean_pooling(model, input_ids, attention_mask, **kwargs):
    last_hidden_state = \
        model(input_ids, attention_mask=attention_mask, output_hidden_states=True, **kwargs).hidden_states[-1]
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    weights_for_non_padding = attention_mask.unsqueeze(-1)
    sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding, dim=1)
    num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
    return sum_embeddings / num_of_none_padding_tokens


def cls_token(model, input_ids, attention_mask, **kwargs):
    return model(input_ids, attention_mask=attention_mask, output_hidden_states=True, **kwargs).last_hidden_state[:, 0]
