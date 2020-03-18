from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import tensorflow as tf
from tensorflow.python.ops import io_ops


class Extractor(ABC):
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    @abstractmethod
    def extract(self, audio):
        pass

    def load_audio_from_file(self, audio_path: Union[str, Path]):
        wav_loader = io_ops.read_file(audio_path)
        audio, sr = tf.audio.decode_wav(wav_loader,
                                        desired_channels=1,
                                        desired_samples=self.sample_rate)

        # delete channel dimension
        # (sampling_rate, 1) -> (sampling_rate,)
        audio = tf.squeeze(audio)

        return audio
