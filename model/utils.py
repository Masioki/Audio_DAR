import os
from typing import Any, List, Dict

import numpy as np
import torch
import torch.nn as nn
from ray import tune
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from transformers import AutoModel, Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback

from config.datasets_config import SLUE_LABEL_2_ID


def slue_label_to_id(batch):
    batch['labels'] = label_to_id(batch['dialog_acts'], SLUE_LABEL_2_ID)
    return batch


def label_to_id(batch, label2id):
    return [one_hot_label_to_id(label, label2id) for label in batch]


def one_hot_label_to_id(labels: str | List[str], label2id: Dict[str, int]):
    if isinstance(labels, str):
        indices = [label2id[labels]]
    else:
        indices = [label2id[label] for label in labels]
    binary = np.zeros(len(label2id))
    binary[indices] = 1
    return binary


def multi_label_compute_metrics(p):
    predictions, labels = p
    predictions = torch.sigmoid(torch.from_numpy(predictions)).cpu().detach().numpy()
    return {
        "precision": precision_score(labels, predictions > 0.5, average='micro'),
        "recall": recall_score(labels, predictions > 0.5, average='micro'),
        "f1-micro": f1_score(labels, predictions > 0.5, average='micro'),
        "f1-macro": f1_score(labels, predictions > 0.5, average='macro'),
        "accuracy": accuracy_score(labels, predictions > 0.5),
    }


def single_label_compute_metrics(p):
    predictions, labels = p
    predictions = torch.softmax(torch.from_numpy(predictions), dim=-1).argmax(dim=-1).cpu().detach().numpy()
    return {
        "precision": precision_score(labels, predictions, average='micro'),
        "recall": recall_score(labels, predictions, average='micro'),
        "f1-micro": f1_score(labels, predictions, average='micro'),
        "f1-macro": f1_score(labels, predictions, average='macro'),
        "accuracy": accuracy_score(labels, predictions),
    }


def ray_hp_space(trial):
    return {
        "weight_decay": tune.loguniform(0.001, 0.1),
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "per_device_train_batch_size": tune.choice([8, 16, 32, 64]),
    }


def train(
        model_provider,
        tokenizer,
        root_path,
        name,
        ds,
        compute_metrics,
        epochs: int = 5,
        report_to="tensorboard",
        lr=4e-5,
        weight_decay=0.01,
        hp_space=None,
        fp16=True,
        tag: str = "default",
        patience: int = 10,
        n_trials: int = 5
):
    model_output_dir = str(os.path.join(root_path, name + "_" + tag))
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        logging_strategy="steps",
        logging_steps=500,
        learning_rate=lr,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        # gradient_checkpointing=True,
        fp16=fp16,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        report_to=report_to,
        ray_scope="all"
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
    if hp_space is None:
        train_res = trainer.train()
    else:
        train_res = trainer.hyperparameter_search(
            direction="maximize",
            backend="ray",
            hp_space=hp_space,
            n_trials=n_trials,
        )
    eval_res = trainer.evaluate(ds["test"])
    trainer.push_to_hub()
    return train_res, eval_res, model_output_dir


def _load_model(name: str, **kwargs) -> Any:
    return AutoModel.from_pretrained(name, **kwargs)


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def wmean_pooling(last_hidden_state, input_ids, attention_mask, **kwargs):
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids).to(last_hidden_state.device)
    weights_for_non_padding = attention_mask * torch.arange(start=1, end=last_hidden_state.shape[1] + 1).unsqueeze(
        0).to(last_hidden_state.device)
    sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
    num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
    return sum_embeddings / num_of_none_padding_tokens


def mean_pooling(last_hidden_state, input_ids, attention_mask, **kwargs):
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids).to(last_hidden_state.device)
    weights_for_non_padding = attention_mask.unsqueeze(-1)
    sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding, dim=1)
    num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
    return sum_embeddings / num_of_none_padding_tokens


def cls_token(last_hidden_state, input_ids, attention_mask, **kwargs):
    return last_hidden_state[:, 0]


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, *args, **kwargs):
        return self.lambd(*args, **kwargs)
