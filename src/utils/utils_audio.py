import pathlib
from typing import Union

import librosa
import librosa.display
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class AudioUtils:
    """ Contains various util functions for handling audio data. """

    @staticmethod
    def load_audio_from_file(path: Union[str, pathlib.Path], sample_rate: int, sample_size: int, stereo_channels: int,
                             to_mono: bool = True):
        """
        Loads audio file from a path, using tensorflow.

        :param path: path to audio file.
        :param sample_rate: sample rate of the audio file.
        :param sample_size: sample size of the audio files.
        :param stereo_channels: channels of the audio file.
        :param to_mono: if the audio should be converted to a mono signal.
        :return: the loaded audio data, shape: (sample_rate * sample_size, channel)
        """
        wav_loader = tf.io.read_file(path)
        # The -32768 to 32767 signed 16-bit values will be scaled to -1.0 to 1.0 in float.
        # desired_samples: sample_rate * sample_size
        audio, sr = tf.audio.decode_wav(wav_loader,
                                        desired_channels=stereo_channels,
                                        desired_samples=sample_rate * sample_size)

        assert audio.shape[-1] == stereo_channels, "Failed to load audio file."

        if to_mono:
            audio = np.mean(audio, axis=1)

        return audio

    @staticmethod
    def visualise_log_mel(anchor, neighbour, opposite):
        """
        Visualises the log mel spectrogram of a triplet of audio files.

        :param anchor: the anchor to visualise.
        :param neighbour: the neighbour to visualise.
        :param opposite: the opposite to visualise.
        :return: None.
        """
        anchor_db = librosa.amplitude_to_db(np.abs(librosa.stft(anchor)), ref=np.max)
        plt.subplot(3, 1, 1)
        plt.title("anchor")
        librosa.display.specshow(anchor_db, y_axis="linear", x_axis="time")

        neighbour_db = librosa.amplitude_to_db(np.abs(librosa.stft(neighbour)), ref=np.max)
        plt.subplot(3, 1, 2)
        plt.title("neighbour")
        librosa.display.specshow(neighbour_db, y_axis="linear", x_axis="time")

        opposite_db = librosa.amplitude_to_db(np.abs(librosa.stft(opposite)), ref=np.max)
        plt.subplot(3, 1, 3)
        plt.title("opposite")
        librosa.display.specshow(opposite_db, y_axis="linear", x_axis="time")

        plt.show()

    @staticmethod
    def visualise_mfcc(anchor, neighbour, opposite, sample_rate):
        """
        Visualises the mfcc of a triplet of audio files.

        :param anchor: the anchor to visualise.
        :param neighbour: the neighbour to visualise.
        :param opposite: the opposite to visualise.
        :param sample_rate: the sampling rate of the audios.
        :return: None.
        """
        mfccs_anchor = librosa.feature.mfcc(y=anchor, sr=sample_rate, n_mfcc=13)
        plt.subplot(3, 1, 1)
        librosa.display.specshow(mfccs_anchor, x_axis="time")

        mfccs_neighbour = librosa.feature.mfcc(y=neighbour, sr=sample_rate, n_mfcc=13)
        plt.subplot(3, 1, 2)
        librosa.display.specshow(mfccs_neighbour, x_axis="time")

        mfccs_opposite = librosa.feature.mfcc(y=opposite, sr=sample_rate, n_mfcc=13)
        plt.subplot(3, 1, 3)
        librosa.display.specshow(mfccs_opposite, x_axis="time")

        plt.show()
