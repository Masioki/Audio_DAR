from typing import List, Dict, Any, Tuple, Callable, Generator

from audio.features import AudioFeature
from dataset_processors.generator import DatasetGenerator
from utils.conversation import Conversation


# TODO: Implement SwdaDatasetGenerator
class SwdaDatasetGenerator(DatasetGenerator):
    def __init__(self, audio_features: List[Callable[[Conversation, int], AudioFeature]], **kwargs):
        super().__init__(audio_features, **kwargs)

    def _prepare(self, split: str) -> None:
        pass

    def _conversation_keys(self, split: str) -> Generator[str, None, None]:
        pass

    def _load_conversation(self, key: str) -> Tuple[Conversation, List[Dict[str, Any]]]:
        pass

    def _get_additional_features(self) -> Dict[str, Any]:
        return {}

    def splits(self) -> List[str]:
        return super().splits()
