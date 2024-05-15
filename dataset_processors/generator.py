import uuid
from typing import List, Any, Tuple, Dict, Callable, Generator

from datasets import Features, Audio, DatasetDict, Dataset

from audio.features import AudioFeature
from config.global_config import SAMPLE_RATE
from config.global_config import log
from utils.conversation import Conversation


class _DatasetGeneratorPickleHack:
    def __init__(self, generator, generator_id=None):
        self.generator = generator
        self.generator_id = (
            generator_id if generator_id is not None else str(uuid.uuid4())
        )

    def __call__(self, *args, **kwargs):
        return self.generator(*args, **kwargs)

    def __reduce__(self):
        return (_DatasetGeneratorPickleHack_raise, (self.generator_id,))


def _DatasetGeneratorPickleHack_raise(*args, **kwargs):
    raise AssertionError("cannot actually unpickle _DatasetGeneratorPickleHack!")


class DatasetGenerator:
    _AUDIO_COLUMN_NAME = "audio"
    _TEXT_COLUMN_NAME = "text"
    _SPEAKER_COLUMN_NAME = "speaker"

    def __init__(self, audio_features: List[Callable[[Conversation, int], AudioFeature]], **kwargs):
        self.audio_features = audio_features

    def _prepare(self, split: str) -> None:
        pass

    def _conversation_keys(self, split: str) -> Generator[str, None, None]:
        raise NotImplementedError

    def _load_conversation(self, key: str) -> Tuple[Conversation, List[Dict[str, Any]]]:
        raise NotImplementedError

    def _get_additional_features(self) -> Dict[str, Any]:
        return {}

    def splits(self) -> List[str]:
        return ["train"]

    def __call__(self, split: str, sample_rate: int):
        log.debug(f"Generating split {split} at sample_rate {sample_rate}")
        self._prepare(split)
        for key in self._conversation_keys(split):
            log.debug(f"Generating conversation {key}")
            conv, features = self._load_conversation(key)
            audio_feats = [f(conv, sample_rate) for f in self.audio_features]
            for i, (speaker, text, wave) in enumerate(conv):
                curr_feats = {af.name: af.get(i) for af in audio_feats if af.name not in features[0].keys()}
                yield {
                    **features[i],
                    **curr_feats,
                    DatasetGenerator._AUDIO_COLUMN_NAME: wave,
                    DatasetGenerator._TEXT_COLUMN_NAME: text,
                    DatasetGenerator._SPEAKER_COLUMN_NAME: speaker
                }

    def to_hf_dataset(self, sample_rate: int) -> DatasetDict[str, Dataset]:
        self._prepare(self.splits()[0])  # TODO: potrzebne do featerów, naprawić to potem
        features = Features({
            DatasetGenerator._AUDIO_COLUMN_NAME: Audio(sampling_rate=SAMPLE_RATE),
            **self._get_additional_features()
        })
        return DatasetDict(
            {split: Dataset.from_generator(_DatasetGeneratorPickleHack(self), features=features,
                                           gen_kwargs={"split": split, "sample_rate": sample_rate}) for split in
             self.splits()})
