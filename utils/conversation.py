from typing import List, Set

import torch


class Conversation:
    def __init__(self, speakers: List[str], texts: List[str], waves: List[torch.Tensor]):
        self.speakers = speakers
        self.texts = texts
        self.waves = waves
        assert len(texts) == len(waves), "Missing corresponding texts to audio"
        assert len(speakers) == len(texts), "Missing corresponding speakers to texts"
        self.speakers_conversations = {}

    def distinct_speakers(self) -> Set[str]:
        return set(self.speakers)

    def extract_speakers_conversation(self, speaker_key: str):
        if speaker_key in self.speakers_conversations:
            return self.speakers_conversations
        indicies = [i for i, key in enumerate(self.speakers) if key == speaker_key]
        conv = Conversation(
            speakers=[speaker_key],
            texts=[self.texts[j] for j in indicies],
            waves=[self.waves[j] for j in indicies]
        )
        self.speakers_conversations[speaker_key] = conv
        return conv

    def __iter__(self, *args, **kwargs):
        for i in range(len(self.speakers)):
            yield self.speakers[i], self.texts[i], self.waves[i]
