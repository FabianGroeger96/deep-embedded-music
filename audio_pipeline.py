import os
import re
import librosa
import librosa.display
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
from typing import Union


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
    def load_audio_from_file(path, sample_rate):
        audio, _ = librosa.load(path, sr=sample_rate)
        # audio = audio.reshape(-1, 1)

        return audio

    @staticmethod
    def visualise_log_mel(anchor, neighbour, opposite):
        D_anchor = librosa.amplitude_to_db(np.abs(librosa.stft(anchor)), ref=np.max)
        plt.subplot(3, 1, 1)
        plt.title("anchor")
        librosa.display.specshow(D_anchor, y_axis='linear', x_axis='time')

        D_neighbour = librosa.amplitude_to_db(np.abs(librosa.stft(neighbour)), ref=np.max)
        plt.subplot(3, 1, 2)
        plt.title("neighbour")
        librosa.display.specshow(D_neighbour, y_axis='linear', x_axis='time')

        D_opposite = librosa.amplitude_to_db(np.abs(librosa.stft(opposite)), ref=np.max)
        plt.subplot(3, 1, 3)
        plt.title("opposite")
        librosa.display.specshow(D_opposite, y_axis='linear', x_axis='time')

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


class AudioPipeline:

    def __init__(self,
                 audio_files_path: Union[str, pathlib.Path],
                 info_file_path: Union[str, pathlib.Path],
                 sample_rate,
                 batch_size,
                 prefetch_batches):
        """
        Initialises the audio pipeline.
        :param audio_files_path: Path to the audio files.
        :param info_file_path: Path to the csv info file, which gives more info about the audio file.
        :param sample_rate: Sample rate with which the audio files are loaded.
        :param batch_size: Batch size of the dataset.
        :param prefetch_batches: Prefetch size of the dataset pipeline.
        """

        self.audio_files_path = Utils.check_if_path_exists(audio_files_path)
        self.info_file_path = Utils.check_if_path_exists(info_file_path)

        self.sample_rate = sample_rate

        files = Utils.get_files_in_path(self.audio_files_path, '.wav')
        if len(files) <= 0:
            raise ValueError("No audio files found in '{}'.".format(self.audio_files_path))
        else:
            print("Found {} audio files.".format(len(files)))

    def generate_samples(self, count):
        i = 0
        audio_info_data_frame = AudioInfoDataFrame(self.audio_files_path, self.info_file_path, self.sample_rate)
        for audio, label, session, node_id, segment in audio_info_data_frame:
            if i == count:
                break

            neighbour = audio_info_data_frame.get_neighbour(label, session, node_id)
            opposite = audio_info_data_frame.get_opposite(label, session, node_id)

            neighbour_audio = Utils.load_audio_from_file(os.path.join(self.audio_files_path, neighbour.sound_file),
                                                         self.sample_rate)
            opposite_audio = Utils.load_audio_from_file(os.path.join(self.audio_files_path, opposite.sound_file),
                                                        self.sample_rate)

            Utils.visualise_log_mel(audio, neighbour_audio, opposite_audio)
            Utils.visualise_mfcc(audio, neighbour_audio, opposite_audio, self.sample_rate)

            i += 1


class AudioInfoDataFrame:
    LABELS = ["absence", "cooking", "dishwashing", "eating", "other", "social_activity", "vacuum_cleaner",
              "watching_tv", "working"]

    def __init__(self,
                 audio_files_path: Union[str, pathlib.Path],
                 info_file_path: Union[str, pathlib.Path],
                 sample_rate: int):
        self.current_index = 0
        self.sample_rate = sample_rate

        self.audio_files_path = Utils.check_if_path_exists(audio_files_path)
        self.info_file_path = Utils.check_if_path_exists(info_file_path)

        self.data_frame = pd.read_csv(self.info_file_path, sep='\t', names=['sound_file',
                                                                            'activity_label',
                                                                            'session'])
        self.data_frame["node_id"] = ""
        self.data_frame["segment"] = ""
        self.data_frame["activity_label"] = self.data_frame["activity_label"].apply(self.LABELS.index)

        self.data_frame["node_id"], self.data_frame["session"], self.data_frame["segment"] = zip(*self.data_frame.apply(
            lambda row: AudioInfoDataFrame.extract_info_from_filename(row["sound_file"]), axis=1))

        print(self.data_frame.head())

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self.current_index > len(self.data_frame):
                raise StopIteration
            else:
                audio_info = self.data_frame.iloc[self.current_index]

                audio_file = audio_info.sound_file
                audio_label = audio_info.activity_label
                audio_session = audio_info.session
                audio_node_id = audio_info.node_id
                audio_segment = audio_info.segment

                print("{0}: audio file: {1}, label: {2}, session: {3}, node id: {4}, segment: {5}".format(
                    self.current_index,
                    audio_file,
                    audio_label,
                    audio_session,
                    audio_node_id,
                    audio_segment))
                audio, _ = librosa.load(os.path.join(self.audio_files_path, audio_file), sr=self.sample_rate)
                # audio = audio.reshape(-1, 1)

                self.current_index += 1

                return audio, audio_label, audio_session, audio_node_id, audio_segment

    @staticmethod
    def extract_info_from_filename(sound_file):
        audio_name_split = sound_file.split("/")
        assert len(audio_name_split) > 1, "Wrong audio file path"
        audio_name_split = audio_name_split[-1].split("_")

        assert len(audio_name_split) == 3, "Wrong audio file name"
        audio_node_id = re.findall(r"\d+", audio_name_split[0])[0]
        audio_session = re.findall(r"\d+", audio_name_split[1])[0]
        audio_segment = re.findall(r"\d+", audio_name_split[2])[0]

        return audio_node_id, audio_session, audio_segment

    def get_neighbour(self, label, session, node):
        filtered_items = self.data_frame
        filtered_items = filtered_items[filtered_items.activity_label == label]
        filtered_items = filtered_items[filtered_items.session != session]
        filtered_items = filtered_items[filtered_items.node_id != node]
        print("Selecting neighbour randomly from {} samples".format(len(filtered_items)))

        neighbour = filtered_items.sample().iloc[0]
        return neighbour

    def get_opposite(self, label, session, node):
        filtered_items = self.data_frame
        filtered_items = filtered_items[filtered_items.activity_label != label]
        print("Selecting opposite randomly from {} samples".format(len(filtered_items)))

        opposite = filtered_items.sample().iloc[0]
        return opposite


INFO_FILE_PATH = "datasets/DCASE18-Task5-development/evaluation_setup/fold1_train.txt"
AUDIO_FILES_PATH = "datasets/DCASE18-Task5-development/"
SAMPLE_RATE = 16000
BATCH_SIZE = 64
PREFETCH_BATCHES = 128

if __name__ == "__main__":
    pipeline = AudioPipeline(AUDIO_FILES_PATH, INFO_FILE_PATH, SAMPLE_RATE, BATCH_SIZE, PREFETCH_BATCHES)
    pipeline.generate_samples(10)
