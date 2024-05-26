from typing import Callable, List

from datasets import DatasetDict
from datasets.data_files import EmptyDatasetError

from config.global_config import SAMPLE_RATE, log
from utils.config import DatasetConfig
from utils.hf import upload, load_ds


def get(config: DatasetConfig, split: str = None, allow_generation: bool = False, force_regenerate: bool = False):
    try:
        log.info(f"Loading dataset {config.repo_path}/{config.repo_name}")
        if force_regenerate:
            log.info(f"Force regenerating")
            return generate(config)
        try:
            return load_ds(config.repo_path, config.repo_name, split)
        except EmptyDatasetError as e:
            log.info(f"Dataset empty, regenerating")
            if allow_generation:
                return generate(config)
            raise Exception(
                "Dataset empty, but regeneration not allowed. Please set allow_generation = True to generate the dataset.")
    except Exception as e:
        log.exception(f"Failed to load dataset {config.repo_path}/{config.repo_name}")
        raise e


def generate(config: DatasetConfig) -> DatasetDict:
    log.info(f"Generating dataset {config.repo_path}/{config.repo_name}")
    ds = config.generator.to_hf_dataset(SAMPLE_RATE)
    try:
        upload(ds, config.repo_path, config.repo_name)
    except Exception:
        log.exception(f"Failed to upload dataset {config.repo_path}/{config.repo_name}")
    return ds


def process(ds, mappers: List[Callable], columns_to_remove: set = {}):
    for mapper in mappers:
        ds = ds.map(mapper, batched=True)
    ds = ds.remove_columns(columns_to_remove)
    return ds


def add_column(config: DatasetConfig, column_name: str, batched_mapper: Callable, save_to_hf: bool = False,
               batch_size: int = 8):
    ds = get(config)
    changed = False
    if type(ds) == DatasetDict:
        for split in ds.keys():
            if column_name not in ds[split].features.keys():
                changed = True
                ds[split] = ds[split].map(lambda batch: {column_name: batched_mapper(batch)}, batched=True,
                                          batch_size=batch_size)
    elif column_name not in ds.features.keys():
        changed = True
        ds = ds.map(lambda batch: {column_name: batched_mapper(batch)}, batched=True, batch_size=batch_size)
    if save_to_hf and changed:
        upload(ds, config.repo_path, config.repo_name)
    return ds


def remove_column(config: DatasetConfig, column_name: str):
    ds = get(config)
    ds = ds.remove_columns([column_name])
    upload(ds, config.repo_path, config.repo_name)
    return ds
