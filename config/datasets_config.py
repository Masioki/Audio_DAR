from audio.features import AudioFeatures
from dataset_processors.slue.generator import SlueDatasetGenerator
from dataset_processors.swda.generator import SwdaDatasetGenerator
from utils.config import DatasetConfig


class Dataset:
    SWDA = DatasetConfig(
        repo_path="Masioki/SWDA-processed",
        audio_features=[AudioFeatures.TEST],
        generator=SwdaDatasetGenerator
    )
    SLUE_TED = DatasetConfig(
        repo_path="Masioki/SLUE-processed",
        repo_name="hvb",
        audio_features=[AudioFeatures.TEST],
        generator=SlueDatasetGenerator,
        generator_kwargs={
            "hf_path": "asapp/slue-phase-2",
            "hf_name": "ted",
            "splits": ["train", "validation", "test"]
        }
    )
    SLUE_VP_NEL = DatasetConfig(
        repo_path="Masioki/SLUE-processed",
        repo_name="vp_nel",
        audio_features=[AudioFeatures.TEST],
        generator=SlueDatasetGenerator,
        generator_kwargs={
            "hf_path": "asapp/slue-phase-2",
            "hf_name": "vp_nel",
            "splits": ["test"]
        }
    )
    SLUE_HVB = DatasetConfig(
        repo_path="Masioki/SLUE-processed",
        repo_name="hvb",
        audio_features=[AudioFeatures.TEST],
        generator=SlueDatasetGenerator,
        generator_kwargs={
            "hf_path": "asapp/slue-phase-2",
            "hf_name": "hvb",
            "splits": ["train", "validation", "test"]
        }
    )
    SLUE_SQA_5 = DatasetConfig(
        repo_path="Masioki/SLUE-processed",
        repo_name="sqa5",
        audio_features=[AudioFeatures.TEST],
        generator=SlueDatasetGenerator,
        generator_kwargs={
            "hf_path": "asapp/slue-phase-2",
            "hf_name": "sqa5",
            "splits": ["train", "validation", "test"]
        }
    )
    # Można tu kombinować z różnymi datasetami
