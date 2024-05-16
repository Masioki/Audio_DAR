from typing import List, Callable, Type

from audio.features import AudioFeature
from dataset_processors.generator import AbstractDatasetGenerator
from utils.conversation import Conversation


class DatasetConfig:
    def __init__(self, repo_path: str, generator: Type[AbstractDatasetGenerator],
                 audio_features: List[Callable[[Conversation, int], AudioFeature]], repo_name: str = None,
                 generator_kwargs=None):
        if generator_kwargs is None:
            generator_kwargs = {}
        self.repo_path = repo_path
        self.repo_name = repo_name
        self.audio_features = audio_features
        self.generator = generator(audio_features, **generator_kwargs)
