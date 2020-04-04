import logging
from typing import Union, Tuple

import numpy as np
import tensorflow as tf

from src.feature_extractor.base_extractor import BaseExtractor
from src.input_pipeline.base_dataset import BaseDataset
from src.utils.params import Params
from src.utils.utils import Utils


class TripletsInputPipeline:
    """Input pipeline to generate triplets."""

    def __init__(self,
                 params: Params,
                 dataset: BaseDataset,
                 log: bool = False):
        """
        Initialises the audio pipeline.
        :param params: parameters of the current experiment.
        :param log: if the pipeline should log details about the data.
        """

        self.dataset_path = Utils.check_if_path_exists(params.dcase_dataset_path)

        self.fold = params.dcase_dataset_fold

        self.sample_rate = params.sample_rate
        self.sample_size = params.sample_size
        self.stereo_channels = params.stereo_channels
        self.to_mono = params.to_mono

        self.batch_size = params.batch_size
        self.prefetch_batches = params.prefetch_batches  # tf.data.experimental.AUTOTUNE / params.prefetch_batches
        self.random_selection_buffer_size = params.random_selection_buffer_size

        self.train_test_split_distribution = params.train_test_split

        self.dataset = dataset

        self.log = log

        self.logger = logging.getLogger(self.__class__.__name__)

        # check if audio path contains *.wav files
        files = Utils.get_files_in_path(self.dataset_path, ".wav")
        if len(files) <= 0:
            raise ValueError("No audio files found in '{}'".format(self.dataset_path))
        else:
            self.logger.info("Found {} audio files".format(len(files)))

    def reinitialise(self):
        self.logger.info("Reinitialising the input pipeline")
        self.dataset.initialise()

    def generate_samples(self, calc_dist: bool = False, trim: bool = True) -> Tuple[np.ndarray, np.ndarray,
                                                                                    np.ndarray, np.ndarray]:
        print_index = 0
        for index, anchor in enumerate(self.dataset):
            triplets, labels = self.dataset.get_triplets(index, calc_dist=calc_dist, trim=trim)

            for triplet, triplet_labels in zip(triplets, labels):
                assert len(triplet) == 3, "Wrong shape of triplets."
                assert len(triplet_labels) == 3, "Wrong shape of triplet labels."

                anchor_audio, neighbour_audio, opposite_audio = triplet
                anchor_label, neighbour_label, opposite_label = triplet_labels

                # make sure audios have the same size
                audio_length = self.sample_size * self.sample_rate
                anchor_audio = anchor_audio[:audio_length]
                neighbour_audio = neighbour_audio[:audio_length]
                opposite_audio = opposite_audio[:audio_length]

                if print_index % 1000 == 0 and self.log:
                    self.logger.debug("Triplet labels, index: {0}, a: {1}, n: {2}, o: {3}".format(index,
                                                                                                  anchor_label,
                                                                                                  neighbour_label,
                                                                                                  opposite_label))
                print_index += 1

                yield anchor_audio, neighbour_audio, opposite_audio, triplet_labels

    def get_dataset(self, feature_extractor: Union[BaseExtractor, None],
                    shuffle: bool = True, calc_dist: bool = False, trim: bool = True):

        if self.to_mono:
            audio_shape = [self.sample_rate * self.sample_size]
        else:
            audio_shape = [self.sample_rate * self.sample_size, self.stereo_channels]

        dataset = tf.data.Dataset.from_generator(self.generate_samples,
                                                 args=[calc_dist, trim],
                                                 output_types=(tf.float32, tf.float32, tf.float32, tf.float32),
                                                 output_shapes=(
                                                     tf.TensorShape(audio_shape),
                                                     tf.TensorShape(audio_shape),
                                                     tf.TensorShape(audio_shape),
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
        dataset = dataset.cache()

        return dataset

    def get_test_dataset(self, extractor):
        test_set = self.dataset.get_test_set(self.stereo_channels, self.to_mono)
        test_set = test_set.to_numpy()

        audio = test_set[:, 0]
        audio = np.transpose(audio)
        audio = tf.convert_to_tensor([audio], dtype=tf.float32)
        audio = tf.squeeze(audio, 0)

        extracted = []
        for a in audio:
            ex = extractor.extract(a)
            extracted.append(ex)

        extracted_tensor = tf.convert_to_tensor(extracted, dtype=tf.float32)

        labels = test_set[:, -1]
        labels = tf.convert_to_tensor([labels], dtype=tf.int32)

        return extracted_tensor, labels
