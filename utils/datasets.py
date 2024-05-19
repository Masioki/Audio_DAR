from typing import Dict, Callable

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


def process(ds, converters: Dict[str, Callable], columns_to_remove: set = {}):
    def mapper(batch):
        for column, converter in converters.items():
            batch[column] = converter(batch[column])

        return batch

    ds = ds.map(mapper, batched=True)
    ds.remove_columns_(columns_to_remove)
    return ds
