from typing import List, Dict, Any, Tuple

from dataset_processors.generator import HfAbstractDatasetGenerator
from utils.conversation import Conversation


# TODO: Implement SwdaDatasetGenerator
class SwdaDatasetGenerator(HfAbstractDatasetGenerator):
    def __init__(self, audio_features, **kwargs):
        super().__init__(audio_features, **kwargs)

    def _load_conversation(self, key: str) -> Tuple[Conversation, List[Dict[str, Any]]]:
        pass
