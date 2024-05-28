import os
from typing import Any, List, Dict, Mapping

import numpy as np
import torch
from ray import tune
from ray.tune.schedulers import MedianStoppingRule
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, Trainer, TrainingArguments, EarlyStoppingCallback

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
        "precision-micro": precision_score(labels, predictions > 0.5, average='micro'),
        "recall-micro": recall_score(labels, predictions > 0.5, average='micro'),
        "recall-macro": recall_score(labels, predictions > 0.5, average='macro'),
        "f1-micro": f1_score(labels, predictions > 0.5, average='micro'),
        "f1-macro": f1_score(labels, predictions > 0.5, average='macro'),
        "f1-weighted": f1_score(labels, predictions > 0.5, average='weighted'),
        "accuracy": accuracy_score(labels, predictions > 0.5),
    }


def single_label_compute_metrics(p):
    predictions, labels = p
    predictions = torch.softmax(torch.from_numpy(predictions), dim=-1).argmax(dim=-1).cpu().detach().numpy()
    return {
        "precision-micro": precision_score(labels, predictions, average='micro'),
        "recall-micro": recall_score(labels, predictions, average='micro'),
        "recall-macro": recall_score(labels, predictions, average='macro'),
        "f1-micro": f1_score(labels, predictions, average='micro'),
        "f1-macro": f1_score(labels, predictions, average='macro'),
        "f1-weighted": f1_score(labels, predictions, average='weighted'),
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
        weight_decay=1e-5,
        hp_space=None,
        fp16=True,
        tag: str = "default",
        patience: int = 10,
        n_trials: int = 5,
        hp_objective: str = "eval_f1-macro",
        metric_for_best_model: str = None,
        val_splits: List[str] = ["validation"],
        eval_splits: List[str] = ["test"],
        skip_memory_metrics=False
):
    model_output_dir = str(os.path.join(root_path, name + "_" + tag))
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        save_total_limit=1,
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
        remove_unused_columns=False,
        metric_for_best_model=metric_for_best_model,
        skip_memory_metrics=skip_memory_metrics,
    )

    if len(val_splits) <= 1:
        test_ds = ds[val_splits[0]]
    else:
        test_ds = {split: ds[split] for split in val_splits}

    trainer = Trainer(
        model_init=model_provider,
        train_dataset=ds["train"],
        eval_dataset=test_ds,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=CustomDataCollator(),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )
    if hp_space is None:
        train_res = trainer.train()
    else:
        return trainer.hyperparameter_search(
            direction="maximize",
            backend="ray",
            hp_space=hp_space,
            n_trials=n_trials,
            compute_objective=lambda metrics: metrics[hp_objective],
            scheduler=MedianStoppingRule(time_attr="training_iteration", metric=hp_objective, grace_period=3,
                                         min_samples_required=3),
        )

    if len(eval_splits) <= 1:
        eval_ds = ds[eval_splits[0]]
    else:
        eval_ds = {split: ds[split] for split in eval_splits}
    eval_res = trainer.evaluate(eval_ds)
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


class CustomDataCollator():
    def __init__(self):
        super().__init__()

    def __call__(self, features):
        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        first = features[0]
        batch = {}
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = pad_sequence([f[k] for f in features], batch_first=True)
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = pad_sequence([torch.tensor(f[k]) for f in features], batch_first=True)
        return batch
