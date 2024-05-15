from typing import Any, Dict, Tuple, List, Callable, Generator

from audio.features import AudioFeature
from dataset_processors.generator import DatasetGenerator
from utils.conversation import Conversation
from utils.hf import load


class SlueDatasetGenerator(DatasetGenerator):
    def __init__(self, audio_features: List[Callable[[Conversation, int], AudioFeature]], hf_path: str, hf_name: str,
                 splits: List[str], **kwargs):
        super().__init__(audio_features, **kwargs)
        self.hf_path = hf_path
        self.hf_name = hf_name
        self.splits_list = splits
        self.ds_iterator = ...
        self.ds = ...
        self.current_conv = ...

    def _prepare(self, split: str) -> None:
        self.ds = load(self.hf_path, self.hf_name, split)
        self.ds_iterator = self.ds.iter(batch_size=1)
        self.current_conv = None

    def _conversation_keys(self, split: str) -> Generator[str, None, None]:
        # chcemy skorzystać z streamingu HF
        # datasety wyglądają na posortowane, więc to podejście oszczędza nam pamięć
        curr_row = next(self.ds_iterator)
        while curr_row is not None:
            new_conv = []
            result = curr_row["issue_id"]
            while result == curr_row["issue_id"]:
                new_conv.append(curr_row)
                curr_row = next(self.ds_iterator, None)
            self.current_conv = new_conv
            yield result

    def _load_conversation(self, key: str) -> Tuple[Conversation, List[Dict[str, Any]]]:
        speakers = [turn.pop("speaker_id") for turn in self.current_conv]
        texts = [turn.pop("text") for turn in self.current_conv]
        audios = [turn.pop("audio") for turn in self.current_conv]
        return Conversation(speakers, texts, audios), self.current_conv

    def _get_additional_features(self) -> Dict[str, Any]:
        return {k: v for k, v in self.ds.features.items() if k not in ["speaker_id", "text", "audio"]}

    def splits(self) -> List[str]:
        return self.splits_list
