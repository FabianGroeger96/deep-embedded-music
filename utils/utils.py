import os
import pathlib
from typing import Union

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


class Utils:

    @staticmethod
    def get_files_in_path(path: Union[str, pathlib.Path], file_extension: str):
        files = []
        for root_dir, dir_names, file_names in os.walk(path):
            # ignore tmp files
            file_names = [f for f in file_names if not f.startswith('.')]
            # find only files with given extension
            file_names = [f for f in file_names if f.endswith(file_extension)]

            for file in file_names:
                files.append(os.path.join(root_dir, file))

        return files

    @staticmethod
    def check_if_path_exists(path: Union[str, pathlib.Path]):
        path = pathlib.Path(path)
        if not path.exists():
            raise ValueError(
                "Path {} is not valid. Please provide an existing path.".format(path))

        return path

    @staticmethod
    def load_audio_from_file(path: Union[str, pathlib.Path], sample_rate: int):
        audio, _ = librosa.load(path, sr=sample_rate)
        audio = audio.reshape(-1, 1)

        return audio

    @staticmethod
    def visualise_log_mel(anchor, neighbour, opposite):
        anchor_db = librosa.amplitude_to_db(np.abs(librosa.stft(anchor)), ref=np.max)
        plt.subplot(3, 1, 1)
        plt.title("anchor")
        librosa.display.specshow(anchor_db, y_axis='linear', x_axis='time')

        neighbour_db = librosa.amplitude_to_db(np.abs(librosa.stft(neighbour)), ref=np.max)
        plt.subplot(3, 1, 2)
        plt.title("neighbour")
        librosa.display.specshow(neighbour_db, y_axis='linear', x_axis='time')

        opposite_db = librosa.amplitude_to_db(np.abs(librosa.stft(opposite)), ref=np.max)
        plt.subplot(3, 1, 3)
        plt.title("opposite")
        librosa.display.specshow(opposite_db, y_axis='linear', x_axis='time')

        plt.show()

    @staticmethod
    def visualise_mfcc(anchor, neighbour, opposite, sample_rate):
        mfccs_anchor = librosa.feature.mfcc(y=anchor, sr=sample_rate, n_mfcc=13)
        plt.subplot(3, 1, 1)
        librosa.display.specshow(mfccs_anchor, x_axis='time')

        mfccs_neighbour = librosa.feature.mfcc(y=neighbour, sr=sample_rate, n_mfcc=13)
        plt.subplot(3, 1, 2)
        librosa.display.specshow(mfccs_neighbour, x_axis='time')

        mfccs_opposite = librosa.feature.mfcc(y=opposite, sr=sample_rate, n_mfcc=13)
        plt.subplot(3, 1, 3)
        librosa.display.specshow(mfccs_opposite, x_axis='time')

        plt.show()
