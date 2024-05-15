from typing import List

import torch

from utils.conversation import Conversation


class AudioFeature:
    def __init__(self, name: str, conversation: Conversation, sample_rate: int):
        self.conversation = conversation
        self.name = name
        self.sample_rate = sample_rate

    def get(self, index: int) -> torch.Tensor:
        raise NotImplementedError


class TestFeature(AudioFeature):
    def __init__(self, conversation: Conversation, return_val: List[int], sample_rate: int):
        super().__init__('test', conversation, sample_rate)
        self.return_val = return_val

    def get(self, index: int) -> torch.Tensor:
        return torch.Tensor(self.return_val)


class AudioFeatures:
    TEST = lambda conv, sr: TestFeature(conv, [1, 2], sr)
