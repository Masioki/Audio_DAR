from datasets import DatasetDict
from datasets.data_files import EmptyDatasetError

from config.global_config import SAMPLE_RATE, log
from utils.config import DatasetConfig
from utils.hf import upload, load


def get(config: DatasetConfig, force_regenerate: bool = False, split: str = None):
    try:
        log.info(f"Loading dataset {config.repo_path}/{config.repo_name}")
        if force_regenerate:
            log.info(f"Force regenerating")
            return generate(config)
        try:
            return load(config.repo_path, config.repo_name, split)
        except EmptyDatasetError as e:
            log.info(f"Dataset empty, regenerating")
            return generate(config)
    except Exception as e:
        log.exception(f"Failed to load dataset {config.repo_path}/{config.repo_name}")
        raise e


def generate(config: DatasetConfig) -> DatasetDict:
    log.info(f"Generating dataset {config.repo_path}/{config.repo_name}")
    ds = config.generator.to_hf_dataset(SAMPLE_RATE)
    upload(ds, config.repo_path, config.repo_name)
    return ds
