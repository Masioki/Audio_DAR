from abc import abstractmethod
from typing import Tuple, Any

import numpy as np
import torch
import torchaudio.functional as F
from datasets import Value, Sequence
from torchaudio.functional.functional import _compute_nccf
from torchaudio.transforms import Vad, MelSpectrogram

from config.global_config import FRAME_SIZE
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


class LogPitchPovWmean(AudioFeature):
    """
    Log pitch with POV weighted mean subtraction over specified window.
    """

    def __init__(self, frame_size_s=0.025, window_size_s=1.5, *args, **kwargs):
        super().__init__('log_pitch_pov', *args, **kwargs)
        self.vad = Vad(self.input_sample_rate)
        self.frame_size_s = frame_size_s
        self.frame_size = int(frame_size_s * self.input_sample_rate)
        self.window_size = int(window_size_s * self.input_sample_rate)

    def _get_hf_dtype(self) -> Any:
        return Sequence(Value(dtype='float32'))

    def get(self, index: int) -> Any:
        utterance_wave = self.conversation.waves[index]
        if type(utterance_wave) is np.ndarray:
            utterance_wave = torch.from_numpy(utterance_wave)
        utterance_pov = self.vad(utterance_wave)
        utterance_pov = torch.cat([torch.zeros(len(utterance_wave) - len(utterance_pov)), utterance_pov])
        pitches = F.detect_pitch_frequency(utterance_wave, sample_rate=self.input_sample_rate,
                                           frame_time=self.frame_size_s, win_length=3)
        return self.normalize_pitch(pitches, utterance_pov)

    def normalize_pitch(self, pitch, pov):
        window_len_frames = int(self.window_size / self.frame_size)
        normalized_pitch = []
        for i in range(pitch.size(-1)):
            start = max(0, i - window_len_frames // 2)
            end = min(pitch.size(-1), i + window_len_frames // 2)
            weights = pov[start:end]
            weighted_pitch = pitch[start:end] * weights
            pov_weighted_mean = torch.sum(weighted_pitch, dim=-1) / (torch.sum(weights) + 1e-6)
            normalized_log_pitch = torch.log(pitch[i] - pov_weighted_mean + 1e-6)
            normalized_pitch.append(normalized_log_pitch)
        return normalized_pitch


class NCCF(AudioFeature):
    def __init__(self, frame_size_s=0.025, *args, **kwargs):
        super().__init__('nccf', *args, **kwargs)
        self.frame_size_s = frame_size_s
        self.frame_size = int(frame_size_s * self.input_sample_rate)

    def _get_hf_dtype(self) -> Any:
        return Sequence(Value(dtype='float32'))

    def get(self, index: int) -> Any:
        waveform = self.conversation.waves[index]
        if type(waveform) is np.ndarray:
            waveform = torch.from_numpy(waveform)
        shape = list(waveform.size())
        waveform = waveform.reshape([-1] + shape[-1:])
        nccf = _compute_nccf(waveform, self.input_sample_rate, self.frame_size_s, 85)
        return


class LogPitchFirstOrderDerivative(AudioFeature):
    def __init__(self, frame_size_s=0.025, *args, **kwargs):
        super().__init__('log_pitch_der', *args, **kwargs)
        self.frame_size_s = frame_size_s
        self.frame_size = int(frame_size_s * self.input_sample_rate)

    def _get_hf_dtype(self) -> Any:
        return Sequence(Value(dtype='float32'))

    def get(self, index: int) -> Any:
        utterance_wave = self.conversation.waves[index]
        if type(utterance_wave) is np.ndarray:
            utterance_wave = torch.from_numpy(utterance_wave)
        pitches = F.detect_pitch_frequency(utterance_wave, sample_rate=self.input_sample_rate,
                                           frame_time=self.frame_size_s, win_length=3)
        pitches = torch.log(pitches)
        der = pitches[1:] - pitches[:-1]
        der = torch.cat([torch.zeros(1), der])
        return der


class LogTotalEnergy(AudioFeature):
    def __init__(self, frame_size_s=0.025, *args, **kwargs):
        super().__init__('log_total_e', *args, **kwargs)
        self.frame_size_s = frame_size_s
        self.frame_size = int(frame_size_s * self.input_sample_rate)

    def _get_hf_dtype(self) -> Any:
        return Sequence(Value(dtype='float32'))

    def get(self, index: int) -> Any:
        utterance_wave = self.conversation.waves[index]
        if type(utterance_wave) is np.ndarray:
            utterance_wave = torch.from_numpy(utterance_wave)
        frames = utterance_wave.unfold(0, self.frame_size, self.frame_size)
        energy = torch.sum(frames ** 2, dim=-1)
        normalized_energies = energy / torch.max(energy)
        return torch.log(normalized_energies + 1e-6)


class LogTotalEnergyLowerBands(AudioFeature):
    def __init__(self, frame_size_s=0.025, bands=40, *args, **kwargs):
        super().__init__('log_total_e_lower_bands', *args, **kwargs)
        self.band = bands // 2
        self.frame_size_s = frame_size_s
        self.frame_size = int(frame_size_s * self.input_sample_rate)
        self.mel = MelSpectrogram(sample_rate=self.input_sample_rate, n_fft=self.frame_size, hop_length=self.frame_size,
                                  n_mels=bands)

    def _get_hf_dtype(self) -> Any:
        return Sequence(Value(dtype='float32'))

    def get(self, index: int) -> Any:
        utterance_wave = self.conversation.waves[index]
        if type(utterance_wave) is np.ndarray:
            utterance_wave = torch.from_numpy(utterance_wave)
        frames = utterance_wave.unfold(0, self.frame_size, self.frame_size)
        energy = torch.sum(frames ** 2, dim=-1)

        mel = self.mel(utterance_wave.float()).transpose(1, 0)
        mel = mel[:frames.shape[0], :self.band]
        mel_energy = torch.sum(mel ** 2, dim=-1)
        normalized_energies = mel_energy / (energy + 1e-6)
        return torch.log(normalized_energies + 1e-6)


class LogTotalEnergyUpperBands(AudioFeature):
    def __init__(self, frame_size_s=0.025, bands=40, *args, **kwargs):
        super().__init__('log_total_e_upper_bands', *args, **kwargs)
        self.band = bands // 2
        self.frame_size_s = frame_size_s
        self.frame_size = int(frame_size_s * self.input_sample_rate)
        self.mel = MelSpectrogram(sample_rate=self.input_sample_rate, n_fft=self.frame_size, hop_length=self.frame_size,
                                  n_mels=bands)

    def _get_hf_dtype(self) -> Any:
        return Sequence(Value(dtype='float32'))

    def get(self, index: int) -> Any:
        utterance_wave = self.conversation.waves[index]
        if type(utterance_wave) is np.ndarray:
            utterance_wave = torch.from_numpy(utterance_wave)
        frames = utterance_wave.unfold(0, self.frame_size, self.frame_size)
        energy = torch.sum(frames ** 2, dim=-1)

        mel = self.mel(utterance_wave.float()).transpose(1, 0)
        mel = mel[:frames.shape[0], :-self.band]
        mel_energy = torch.sum(mel ** 2, dim=-1)
        normalized_energies = mel_energy / (energy + 1e-6)
        return torch.log(normalized_energies + 1e-6)


class AudioFeatures:
    LOG_PITCH_POV = lambda conv, in_sr, sr: LogPitchPovWmean(FRAME_SIZE, 1.5, conv, in_sr, sr)
    LOG_PITCH_DER = lambda conv, in_sr, sr: LogPitchFirstOrderDerivative(FRAME_SIZE, conv, in_sr, sr)
    LOG_TOTAL_E_LOWER_BANDS = lambda conv, in_sr, sr: LogTotalEnergyLowerBands(FRAME_SIZE, 40, conv, in_sr, sr)
    LOG_TOTAL_E_UPPER_BANDS = lambda conv, in_sr, sr: LogTotalEnergyUpperBands(FRAME_SIZE, 40, conv, in_sr, sr)
    LOG_TOTAL_E = lambda conv, in_sr, sr: LogTotalEnergy(FRAME_SIZE, conv, in_sr, sr)
    NCCF = lambda conv, in_sr, sr: LogPitchPovWmean(FRAME_SIZE, conv, in_sr, sr)  # TODO
