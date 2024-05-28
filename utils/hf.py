from typing import Any

import huggingface_hub as hub
from datasets import Dataset, load_dataset, DatasetDict
from transformers import AutoModel

from config.global_config import HF_CONFIG, log


def login_to_hf():
    try:
        log.info(f"Logging in to HF")
        hub.login(token=HF_CONFIG["token"], write_permission=HF_CONFIG["write_permission"])
    except Exception:
        log.exception(f"Failed to login to HF")


def upload(ds: Dataset | DatasetDict, path: str, name: str = "default", split: str = None):
    login_to_hf()
    log.info(f"Uploading dataset {path}/{name}/{split} to HF")
    if type(ds) == DatasetDict:
        ds.push_to_hub(path, name)
        return
    ds.push_to_hub(path, name, split=split)


def load_ds(path: str, name: str = None, splits=None) -> Dataset | DatasetDict:
    log.debug(f"Loading dataset {path}/{name}/{splits} from HF")
    if splits and type(splits) == list:
        return DatasetDict({split: load_dataset(path, name=name, split=split) for split in splits})
    return load_dataset(path, name=name, split=splits)


def load_model(name: str, **kwargs) -> Any:
    log.debug(f"Loading model {name} from HF")
    login_to_hf()
    return AutoModel.from_pretrained(name, **kwargs)
