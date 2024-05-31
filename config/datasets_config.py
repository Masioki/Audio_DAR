from audio.features import AudioFeatures
from dataset_processors.generator import OrderedHfDatasetGenerator
from dataset_processors.swda.generator import SwdaDatasetGenerator
from utils.config import DatasetConfig

SLUE_ID_2_LABEL = {
    0: "question_check",
    1: "question_repeat",
    2: "question_general",
    3: "answer_agree",
    4: "answer_dis",
    5: "answer_general",
    6: "apology",
    7: "thanks",
    8: "acknowledge",
    9: "statement_open",
    10: "statement_close",
    11: "statement_problem",
    12: "statement_instruct",
    13: "statement_general",
    14: "backchannel",
    15: "disfluency",
    16: "self",
    17: "other"
}

SLUE_LABEL_2_ID = {v: k for k, v in SLUE_ID_2_LABEL.items()}


class Dataset:
    SWDA = DatasetConfig(
        repo_path="Masioki/SWDA-processed",
        audio_features=[AudioFeatures.LOG_PITCH_POV],
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
    SLUE_HVB = DatasetConfig(
        repo_path="Masioki/SLUE-processed",
        repo_name="hvb",
        audio_features=[
            AudioFeatures.LOG_PITCH_POV,
            AudioFeatures.LOG_PITCH_DER,
            AudioFeatures.LOG_TOTAL_E,
            AudioFeatures.LOG_TOTAL_E_LOWER_BANDS,
            AudioFeatures.LOG_TOTAL_E_UPPER_BANDS
        ],
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
    SLUE_HVB_PROCESSED = DatasetConfig(
        repo_path="Masioki/SLUE-processed",
        repo_name="hvb",
        audio_features=[
            AudioFeatures.LOG_PITCH_POV,
            AudioFeatures.LOG_PITCH_DER,
            AudioFeatures.LOG_TOTAL_E,
            AudioFeatures.LOG_TOTAL_E_LOWER_BANDS,
            AudioFeatures.LOG_TOTAL_E_UPPER_BANDS
        ],
        generator=OrderedHfDatasetGenerator,
        generator_kwargs={
            "hf_path": "Masioki/SLUE-processed",
            "hf_name": "hvb",
            "splits_config": ["train", "validation", "test", "asr_train", "asr_validation", "asr_test"],
            "audio_id": "audio",
            "speaker_id": "speaker",
            "text_id": "text",
            "conv_id": "conversation"
        }
    )
    # Można tu kombinować z różnymi datasetami
