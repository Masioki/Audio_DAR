from audio.features import AudioFeatures
from dataset_processors.generator import OrderedHfDatasetGenerator
from dataset_processors.swda.generator import SwdaDatasetGenerator
from utils.config import DatasetConfig


class Dataset:
    SWDA = DatasetConfig(
        repo_path="Masioki/SWDA-processed",
        audio_features=[AudioFeatures.TEST],
        generator=SwdaDatasetGenerator,
        generator_kwargs={
            "hf_path": "swda",
            "hf_name": None,
            "splits_config": ["train", "validation", "test"],
            "audio_id": None,
            "speaker_id": "caller",
            "text_id": "text",
            "conv_id": "conversation_no"
        }
    )
    SLUE_TED = DatasetConfig(  # TODO: parametry
        repo_path="Masioki/SLUE-processed",
        repo_name="ted",
        audio_features=[AudioFeatures.TEST],
        generator=OrderedHfDatasetGenerator,
        generator_kwargs={
            "hf_path": "asapp/slue-phase-2",
            "hf_name": "ted",
            "splits_config": ["train", "validation", "test"],
            "audio_id": "audio",
            "speaker_id": "speaker_id",
            "text_id": "text",
            "conv_id": "issue_id"
        }
    )
    SLUE_VP_NEL = DatasetConfig(
        repo_path="Masioki/SLUE-processed",
        repo_name="vp_nel",
        audio_features=[AudioFeatures.TEST],
        generator=OrderedHfDatasetGenerator,
        generator_kwargs={
            "hf_path": "asapp/slue-phase-2",
            "hf_name": "vp_nel",
            "splits_config": ["train", "validation", "test"],
            "audio_id": "audio",
            "speaker_id": "speaker_id",
            "text_id": "text",
            "conv_id": "id"
        }
    )
    SLUE_HVB = DatasetConfig(
        repo_path="Masioki/SLUE-processed",
        repo_name="hvb",
        audio_features=[AudioFeatures.TEST],
        generator=OrderedHfDatasetGenerator,
        generator_kwargs={
            "hf_path": "asapp/slue-phase-2",
            "hf_name": "hvb",
            "splits_config": ["train", "validation", "test"],
            "audio_id": "audio",
            "speaker_id": "speaker_id",
            "text_id": "text",
            "conv_id": "issue_id"
        }
    )
    SLUE_SQA_5 = DatasetConfig(  # TODO: parametry
        repo_path="Masioki/SLUE-processed",
        repo_name="sqa5",
        audio_features=[AudioFeatures.TEST],
        generator=OrderedHfDatasetGenerator,
        generator_kwargs={
            "hf_path": "asapp/slue-phase-2",
            "hf_name": "sqa5",
            "splits_config": ["train", "validation", "test"],
            "audio_id": "audio",
            "speaker_id": "speaker_id",
            "text_id": "text",
            "conv_id": "issue_id"
        }
    )
    # Można tu kombinować z różnymi datasetami
