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


def upload(ds: Dataset | DatasetDict, path: str, name: str = "default"):
    login_to_hf()
    log.info(f"Uploading dataset {path}/{name} to HF")
    ds.push_to_hub(path, name)


def load_ds(path: str, name: str = None, split: str = None) -> Dataset | DatasetDict:
    log.debug(f"Loading dataset {path}/{name}/{split} from HF")
    return load_dataset(path, name=name, split=split)


def load_model(name: str, **kwargs) -> Any:
    log.debug(f"Loading model {name} from HF")
    login_to_hf()
    return AutoModel.from_pretrained(name, **kwargs)
