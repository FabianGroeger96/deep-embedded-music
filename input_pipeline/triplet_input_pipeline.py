import os
import pathlib
from typing import Union

import tensorflow as tf

from input_pipeline.dcase_data_frame import DCASEDataFrame
from utils.utils import Utils


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

        files = Utils.get_files_in_path(self.audio_files_path, '.wav')
        if len(files) <= 0:
            raise ValueError("No audio files found in '{}'.".format(self.audio_files_path))
        else:
            print("Found {} audio files.".format(len(files)))

    def generate_samples(self):
        audio_info_data_frame = DCASEDataFrame(self.audio_files_path, self.info_file_path, self.sample_rate)
        for audio, label, session, node_id, segment in audio_info_data_frame:
            neighbour = audio_info_data_frame.get_neighbour(label, session, node_id)
            opposite = audio_info_data_frame.get_opposite(label)

            neighbour_audio = Utils.load_audio_from_file(os.path.join(self.audio_files_path, neighbour.sound_file),
                                                         self.sample_rate)
            opposite_audio = Utils.load_audio_from_file(os.path.join(self.audio_files_path, opposite.sound_file),
                                                        self.sample_rate)

            yield audio, neighbour_audio, opposite_audio

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

    def generate_samples_visualise(self, count):
        i = 0
        audio_info_data_frame = DCASEDataFrame(self.audio_files_path, self.info_file_path, self.sample_rate)
        for audio, label, session, node_id, segment in audio_info_data_frame:
            if i == count:
                break

            neighbour = audio_info_data_frame.get_neighbour(label, session, node_id)
            opposite = audio_info_data_frame.get_opposite(label)

            neighbour_audio = Utils.load_audio_from_file(os.path.join(self.audio_files_path, neighbour.sound_file),
                                                         self.sample_rate)
            opposite_audio = Utils.load_audio_from_file(os.path.join(self.audio_files_path, opposite.sound_file),
                                                        self.sample_rate)

            Utils.visualise_log_mel(audio, neighbour_audio, opposite_audio)
            Utils.visualise_mfcc(audio, neighbour_audio, opposite_audio, self.sample_rate)

            i += 1
