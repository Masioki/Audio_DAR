import uuid
from abc import abstractmethod
from typing import List, Any, Tuple, Dict, Callable, Generator

from datasets import Features, Audio, DatasetDict, Dataset, Value

from audio.features import AudioFeature
from config.global_config import AUDIO_COLUMN_NAME, TEXT_COLUMN_NAME, SPEAKER_COLUMN_NAME, CONVERSATION_COLUMN_NAME
from config.global_config import SAMPLE_RATE
from config.global_config import log
from utils.conversation import Conversation
from utils.hf import load


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


class AbstractDatasetGenerator:

    def __init__(self, audio_features: List[Callable[[Conversation, int, int], AudioFeature]],
                 splits_config: List[str], **kwargs):
        self.audio_features = audio_features
        self.splits_config = splits_config
        self.audio_hf_features = None

    def _prepare(self, split: str) -> None:
        pass

    @abstractmethod
    def _conversation_keys(self, split: str) -> Generator[str, None, None]:
        raise NotImplementedError

    @abstractmethod
    def _load_conversation(self, key: str) -> Tuple[Conversation, List[Dict[str, Any]]]:
        raise NotImplementedError

    @abstractmethod
    def _input_sample_rate(self) -> int:
        raise NotImplementedError

    def _get_additional_features(self) -> Dict[str, Any]:
        return {}

    def __call__(self, split: str, sample_rate: int) -> Generator[Dict[str, Any], None, None]:
        log.debug(f"Generating split {split} at sample_rate {sample_rate}")
        self._prepare(split)
        for key in self._conversation_keys(split):
            log.debug(f"Generating conversation {key}")
            conv, features = self._load_conversation(key)
            audio_feats = [f(conv, self._input_sample_rate(), sample_rate) for f in self.audio_features]
            for i, (speaker, text, wave) in enumerate(conv):
                curr_feats = {af.name: af.get(i) for af in audio_feats if af.name not in features[0].keys()}
                yield {
                    **features[i],
                    **curr_feats,
                    AUDIO_COLUMN_NAME: {
                        'path': f'conversation_{key}_turn_{i}.wav',
                        'array': wave,
                        'sampling_rate': self._input_sample_rate()
                    },
                    TEXT_COLUMN_NAME: text,
                    SPEAKER_COLUMN_NAME: speaker,
                    CONVERSATION_COLUMN_NAME: key
                }

    def to_hf_dataset(self, sample_rate: int) -> DatasetDict[str, Dataset]:
        # TODO: potrzebne do featerów, naprawić to potem
        self._prepare(self.splits_config[0])
        self.audio_hf_features = [f(Conversation([], [], []), sample_rate, sample_rate) for f in self.audio_features]
        self.audio_hf_features = [af.get_hf_feature() for af in self.audio_hf_features]
        self.audio_hf_features = {k: v for k, v in self.audio_hf_features}
        ##

        features = Features({
            AUDIO_COLUMN_NAME: Audio(sampling_rate=SAMPLE_RATE),
            TEXT_COLUMN_NAME: Value('string'),
            SPEAKER_COLUMN_NAME: Value('string'),
            CONVERSATION_COLUMN_NAME: Value('string'),
            **self._get_additional_features(),
            **self.audio_hf_features
        })
        # TODO: from_generator nie działa dla nie pickowalnych obiektów
        # _DatasetGeneratorPickleHack(self)
        # return DatasetDict({
        #     split: Dataset.from_generator(self,
        #                                   features=features,
        #                                   gen_kwargs={"split": split, "sample_rate": sample_rate,
        #                                               "calculate_features": gen_feats})
        #     for split, gen_feats in self.splits_config.items()
        # })
        return DatasetDict({
            split: Dataset.from_list([i for i in self(split, sample_rate)], features=features)
            for split in self.splits_config
        })


class HfAbstractDatasetGenerator(AbstractDatasetGenerator):
    def __init__(
            self,
            audio_features: List[Callable[[Conversation, int, int], AudioFeature]],
            hf_path: str,
            hf_name: str,
            splits_config: List[str],
            conv_id: str,
            audio_id: str,
            text_id: str,
            speaker_id: str,
            **kwargs
    ):
        super().__init__(audio_features, splits_config, **kwargs)
        self.hf_path = hf_path
        self.hf_name = hf_name
        self.conv_id = conv_id
        self.audio_id = audio_id
        self.text_id = text_id
        self.speaker_id = speaker_id
        self.ds_iterator = ...
        self.ds = ...

    def _input_sample_rate(self) -> int:
        return self.ds.features[self.audio_id].sampling_rate

    def _prepare(self, split: str) -> None:
        self.ds = load(self.hf_path, self.hf_name, split)
        self.ds_iterator = self.ds.iter(batch_size=1)

    def _conversation_keys(self, split: str) -> Generator[str, None, None]:
        for k in set(self.ds[self.conv_id]):
            yield k

    def _get_additional_features(self) -> Dict[str, Any]:
        return {k: v for k, v in self.ds.features.items() if
                k not in [self.conv_id, self.speaker_id, self.text_id, self.audio_id]}

    @abstractmethod
    def _load_conversation(self, key: str) -> Tuple[Conversation, List[Dict[str, Any]]]:
        raise NotImplementedError


class OrderedHfDatasetGenerator(HfAbstractDatasetGenerator):
    def __init__(self, audio_features: List[Callable[[Conversation, int, int], AudioFeature]], **kwargs):
        super().__init__(audio_features, **kwargs)
        self.current_conv = ...

    def _prepare(self, split: str) -> None:
        super()._prepare(split)
        self.current_conv = None

    def _conversation_keys(self, split: str) -> Generator[str, None, None]:
        # chcemy skorzystać z streamingu HF
        # datasety wyglądają na posortowane, więc to podejście oszczędza nam pamięć
        curr_row = next(self.ds_iterator)
        while curr_row is not None:
            new_conv = []
            result = curr_row[self.conv_id][0]
            while curr_row is not None and result == curr_row[self.conv_id][0]:
                new_conv.append(curr_row)
                curr_row = next(self.ds_iterator, None)
            self.current_conv = new_conv
            yield result

    def _load_conversation(self, key: str) -> Tuple[Conversation, List[Dict[str, Any]]]:
        unbatched = []
        for turn in self.current_conv:
            unbatched.append({k: v[0] for k, v in turn.items()})
        self.current_conv = unbatched
        speakers = [turn.pop(self.speaker_id) for turn in self.current_conv]
        texts = [turn.pop(self.text_id) for turn in self.current_conv]
        audios = [turn.pop(self.audio_id)['array'] for turn in self.current_conv]
        convs = [turn.pop(self.conv_id) for turn in self.current_conv]
        return Conversation(speakers, texts, audios), self.current_conv
