import logging
import os
import pathlib
from typing import Union

import numpy as np
import tensorflow as tf

from feature_extractor.extractor import Extractor
from input_pipeline.dcase_data_frame import DCASEDataFrame
from utils.utils import Utils


class TripletsInputPipeline:
    """Input pipeline to generate triplets."""

    def __init__(self,
                 dataset_path: Union[str, pathlib.Path],
                 fold: int,
                 sample_rate: int,
                 sample_size: int,
                 batch_size: int,
                 prefetch_batches: int,
                 random_selection_buffer_size: int,
                 input_processing_n_threads: int = 16,
                 stereo_channels: int = 4):
        """
        Initialises the audio pipeline.
        :param dataset_path: Path to the dataset.
        :param sample_rate: Sample rate with which the audio files are loaded.
        :param batch_size: Batch size of the dataset.
        :param prefetch_batches: Prefetch size of the dataset pipeline.
        """
        self.dataset_path = Utils.check_if_path_exists(dataset_path)

        self.fold = fold

        self.sample_rate = sample_rate
        self.sample_size = sample_size

        self.batch_size = batch_size
        self.prefetch_batches = prefetch_batches
        self.random_selection_buffer_size = random_selection_buffer_size
        self.input_processing_n_threads = input_processing_n_threads

        self.stereo_channels = stereo_channels

        self.logger = logging.getLogger(self.__class__.__name__)

        # check if audio path contains *.wav files
        files = Utils.get_files_in_path(self.dataset_path, '.wav')
        if len(files) <= 0:
            raise ValueError("No audio files found in '{}'.".format(self.dataset_path))
        else:
            self.logger.debug("Found {} audio files.".format(len(files)))

    def generate_samples(self, calc_dist: bool = False) -> [np.ndarray, np.ndarray, np.ndarray]:
        audio_info_data_frame = DCASEDataFrame(self.dataset_path, fold=self.fold, sample_rate=self.sample_rate)
        for index, anchor in enumerate(audio_info_data_frame):
            try:
                neighbour, neighbour_dist = audio_info_data_frame.get_neighbour(index, calc_dist=calc_dist)
                opposite, opposite_dist = audio_info_data_frame.get_opposite(index, calc_dist=calc_dist)

                # distance to differ between hard / easy triplet
                if calc_dist:
                    if neighbour_dist > opposite_dist:
                        self.logger.debug("Dist opposite smaller than dist neighbour --> hard triplet")
                    else:
                        self.logger.debug("Dist opposite bigger than dist neighbour --> easy triplet")

            except ValueError as err:
                self.logger.debug("Error during triplet computation: {}".format(err))
                continue
            except:
                break

            # load audio files from anchor
            anchor_path = os.path.join(self.dataset_path, anchor.file_name)
            anchor_audio = Utils.load_audio_from_file(anchor_path, self.sample_rate, self.stereo_channels)
            # load audio files from neighbour
            neighbour_path = os.path.join(self.dataset_path, neighbour.file_name)
            neighbour_audio = Utils.load_audio_from_file(neighbour_path, self.sample_rate, self.stereo_channels)
            # load audio files from opposite
            opposite_path = os.path.join(self.dataset_path, opposite.file_name)
            opposite_audio = Utils.load_audio_from_file(opposite_path, self.sample_rate, self.stereo_channels)

            # make sure audios have the same size
            audio_length = self.sample_size * self.sample_rate
            anchor_audio = anchor_audio[:audio_length]
            neighbour_audio = neighbour_audio[:audio_length]
            opposite_audio = opposite_audio[:audio_length]

            # create tuple of triplet labels
            triplet_labels = np.stack((anchor.label, neighbour.label, opposite.label), axis=0)
            self.logger.debug("Triplet labels, a: {0}, n: {1}, o: {2}".format(anchor.label,
                                                                              neighbour.label,
                                                                              opposite.label))

            yield anchor_audio, neighbour_audio, opposite_audio, triplet_labels

    def get_dataset(self, feature_extractor: Union[Extractor, None], shuffle: bool = True, calc_dist: bool = False):
        # why not sample rate * sample_size?
        dataset = tf.data.Dataset.from_generator(self.generate_samples,
                                                 args=[calc_dist],
                                                 output_types=(tf.float32, tf.float32, tf.float32, tf.float32),
                                                 output_shapes=(
                                                     tf.TensorShape([self.sample_rate, self.stereo_channels]),
                                                     tf.TensorShape([self.sample_rate, self.stereo_channels]),
                                                     tf.TensorShape([self.sample_rate, self.stereo_channels]),
                                                     tf.TensorShape([3])))

        # extract features from dataset
        if feature_extractor is not None:
            dataset = dataset.map(lambda a, n, o, labels: (
                feature_extractor.extract(a),
                feature_extractor.extract(n),
                feature_extractor.extract(o),
                labels))

        if shuffle:
            # buffer size defines from how much elements are in the buffer, from which then will get shuffled
            dataset = dataset.shuffle(buffer_size=self.random_selection_buffer_size)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.prefetch_batches)

        return dataset
