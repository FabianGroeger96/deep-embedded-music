import logging
import os
import pathlib
from typing import Union

import tensorflow as tf

from input_pipeline.dcase_data_frame import DCASEDataFrame
from utils.utils import Utils, set_logger


class TripletsInputPipeline:
    """Input pipeline to generate triplets."""

    def __init__(self,
                 audio_files_path: Union[str, pathlib.Path],
                 info_file_path: Union[str, pathlib.Path],
                 sample_rate: int,
                 sample_size: int,
                 batch_size: int,
                 prefetch_batches: int,
                 input_processing_buffer_size: int):
        """
        Initialises the audio pipeline.
        :param audio_files_path: Path to the audio files.
        :param info_file_path: Path to the csv info file, which gives more info about the audio file.
        :param sample_rate: Sample rate with which the audio files are loaded.
        :param batch_size: Batch size of the dataset.
        :param prefetch_batches: Prefetch size of the dataset pipeline.
        """
        self.batch_size = batch_size
        self.prefetch_batches = prefetch_batches
        self.input_processing_buffer_size = input_processing_buffer_size

        self.audio_files_path = Utils.check_if_path_exists(audio_files_path)
        self.info_file_path = Utils.check_if_path_exists(info_file_path)

        self.sample_rate = sample_rate
        self.sample_size = sample_size

        self.logger = logging.getLogger()

        files = Utils.get_files_in_path(self.audio_files_path, '.wav')
        if len(files) <= 0:
            raise ValueError("No audio files found in '{}'.".format(self.audio_files_path))
        else:
            self.logger.debug("Found {} audio files.".format(len(files)))

    def generate_samples(self):
        audio_info_data_frame = DCASEDataFrame(self.audio_files_path, self.info_file_path, self.sample_rate)
        for index, audio_series in enumerate(audio_info_data_frame):
            audio, label, session, node_id, segment = audio_series

            neighbour, neighbour_dist = audio_info_data_frame.get_neighbour(index)
            opposite, opposite_dist = audio_info_data_frame.get_opposite(index)

            neighbour_audio = Utils.load_audio_from_file(os.path.join(self.audio_files_path, neighbour.sound_file),
                                                         self.sample_rate)
            opposite_audio = Utils.load_audio_from_file(os.path.join(self.audio_files_path, opposite.sound_file),
                                                        self.sample_rate)

            # yield audio, neighbour_audio, opposite_audio
            # break

    def get_dataset(self, is_training: bool = True):
        dataset = tf.data.Dataset.from_generator(self.generate_samples,
                                                 output_types=(tf.float32, tf.float32, tf.float32),
                                                 output_shapes=(
                                                     tf.TensorShape([self.sample_rate * self.sample_size, 1]),
                                                     tf.TensorShape([self.sample_rate * self.sample_size, 1]),
                                                     tf.TensorShape([self.sample_rate * self.sample_size, 1])))
        if is_training:
            dataset = dataset.shuffle(buffer_size=self.input_processing_buffer_size)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.prefetch_batches)

        return dataset
