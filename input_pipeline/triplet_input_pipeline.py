import logging
import os
import pathlib
from enum import Enum
from typing import Union

import tensorflow as tf
import numpy as np

from input_pipeline.dcase_data_frame import DCASEDataFrame
from utils.utils import Utils


class DatasetMode(Enum):
    train = 1
    valid = 2
    test = 3


class TripletsInputPipeline:
    """Input pipeline to generate triplets."""

    def __init__(self,
                 audio_files_path: Union[str, pathlib.Path],
                 info_file_path: Union[str, pathlib.Path],
                 sample_rate: int,
                 sample_size: int,
                 batch_size: int,
                 prefetch_batches: int,
                 input_processing_buffer_size: int,
                 category_cardinality: int):
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
        self.sample_size = sample_size

        self.batch_size = batch_size
        self.prefetch_batches = prefetch_batches
        self.input_processing_buffer_size = input_processing_buffer_size

        self.category_cardinality = category_cardinality

        self.logger = logging.getLogger()

        # check if audio path contains *.wav files
        files = Utils.get_files_in_path(self.audio_files_path, '.wav')
        if len(files) <= 0:
            raise ValueError("No audio files found in '{}'.".format(self.audio_files_path))
        else:
            self.logger.debug("Found {} audio files.".format(len(files)))

    def generate_samples(self) -> [np.ndarray, np.ndarray, np.ndarray]:
        audio_info_data_frame = DCASEDataFrame(self.audio_files_path, self.info_file_path, self.sample_rate)
        for index, audio_series in enumerate(audio_info_data_frame):
            audio, label, session, node_id, segment = audio_series

            try:
                neighbour, neighbour_dist = audio_info_data_frame.get_neighbour(index)
                opposite, opposite_dist = audio_info_data_frame.get_opposite(index)
            except ValueError as err:
                self.logger.debug("Error during triplet computation: {}".format(err))
                continue
            except:
                break

            neighbour_audio = Utils.load_audio_from_file(os.path.join(self.audio_files_path, neighbour.sound_file),
                                                         self.sample_rate)
            opposite_audio = Utils.load_audio_from_file(os.path.join(self.audio_files_path, opposite.sound_file),
                                                        self.sample_rate)

            triplet_labels = np.stack((label, neighbour.activity_label, opposite.activity_label), axis=0)
            yield audio, neighbour_audio, opposite_audio, triplet_labels

    def get_dataset(self, mode: DatasetMode = DatasetMode.test, shuffle: bool = True):
        dataset = None
        if mode == DatasetMode.test:
            dataset = tf.data.Dataset.from_generator(self.generate_samples,
                                                     output_types=(tf.float32, tf.float32, tf.float32,
                                                                   tf.float32),
                                                     output_shapes=(
                                                         tf.TensorShape([self.sample_rate * self.sample_size, 1]),
                                                         tf.TensorShape([self.sample_rate * self.sample_size, 1]),
                                                         tf.TensorShape([self.sample_rate * self.sample_size, 1]),
                                                         tf.TensorShape([3])))  # triplet labels
        elif mode == DatasetMode.valid:
            dataset = tf.data.Dataset.from_generator(self.generate_samples,
                                                     output_types=(tf.float32, tf.float32),
                                                     output_shapes=(
                                                         tf.TensorShape([self.sample_rate * self.sample_size, 1]),
                                                         tf.TensorShape([1])))  # label of audio
        elif mode == DatasetMode.test:
            dataset = tf.data.Dataset.from_generator(self.generate_samples,
                                                     output_types=tf.float32,
                                                     output_shapes=(
                                                         tf.TensorShape([self.sample_rate * self.sample_size, 1])))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.input_processing_buffer_size)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.prefetch_batches)

        return dataset
