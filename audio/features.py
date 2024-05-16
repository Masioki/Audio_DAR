from abc import abstractmethod
from typing import List, Tuple, Any

from datasets import Value, Sequence

from utils.conversation import Conversation


class AudioFeature:
    def __init__(self, name: str, conversation: Conversation, input_sample_rate: int, target_sample_rate: int):
        self.conversation = conversation
        self.name = name
        self.input_sample_rate = input_sample_rate
        self.target_sample_rate = target_sample_rate

    def get_hf_feature(self) -> Tuple[str, Any]:
        return self.name, self._get_hf_dtype()

    @abstractmethod
    def _get_hf_dtype(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get(self, index: int) -> Any:
        raise NotImplementedError


class TestFeature(AudioFeature):
    def __init__(self, return_val: List[int], *args, **kwargs):
        super().__init__('test', *args, **kwargs)
        self.return_val = return_val

    def _get_hf_dtype(self) -> Any:
        return Sequence(feature=Value(dtype='int32'))

    def get(self, index: int) -> Any:
        return self.return_val


class AudioFeatures:
    TEST = lambda conv, in_sr, sr: TestFeature([1, 2], conv, in_sr, sr)
