import logging
import os
from typing import Union

import numpy as np
import tensorflow as tf

from src.feature_extractor.extractor import Extractor
from src.input_pipeline.dcase_dataset import DCASEDataset
from src.utils.audio_utils import AudioUtils
from src.utils.params import Params
from src.utils.utils import Utils


class TripletsInputPipeline:
    """Input pipeline to generate triplets."""

    def __init__(self,
                 params: Params,
                 log: bool = False):
        """
        Initialises the audio pipeline.
        :param params: parameters of the current experiment.
        :param log: if the pipeline should log details about the data.
        """

        self.dataset_path = Utils.check_if_path_exists(params.audio_files_path)

        self.fold = params.fold

        self.sample_rate = params.sample_rate
        self.sample_size = params.sample_size

        self.batch_size = params.batch_size
        self.prefetch_batches = tf.data.experimental.AUTOTUNE  # params.prefetch_batches
        self.random_selection_buffer_size = params.random_selection_buffer_size

        self.stereo_channels = params.stereo_channels
        self.to_mono = params.to_mono

        self.train_test_split_distribution = params.train_test_split

        self.log = log

        self.logger = logging.getLogger(self.__class__.__name__)

        # check if audio path contains *.wav files
        files = Utils.get_files_in_path(self.dataset_path, ".wav")
        if len(files) <= 0:
            raise ValueError("No audio files found in '{}'".format(self.dataset_path))
        else:
            self.logger.info("Found {} audio files".format(len(files)))

        self.audio_info_df = DCASEDataset(self.dataset_path, fold=self.fold, sample_rate=self.sample_rate,
                                          train_test_split_distribution=self.train_test_split_distribution)

    def reinitialise(self):
        self.logger.info("Reinitialising the input pipeline")
        self.audio_info_df = DCASEDataset(self.dataset_path, fold=self.fold, sample_rate=self.sample_rate,
                                          train_test_split_distribution=self.train_test_split_distribution)

    def generate_samples(self, calc_dist: bool = False) -> [np.ndarray, np.ndarray, np.ndarray]:
        for index, anchor in enumerate(self.audio_info_df):
            try:
                neighbour, neighbour_dist = self.audio_info_df.get_neighbour(index, calc_dist=calc_dist)
                opposite, opposite_dist = self.audio_info_df.get_opposite(index, calc_dist=calc_dist)

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
            anchor_audio = AudioUtils.load_audio_from_file(anchor_path, self.sample_rate, self.stereo_channels,
                                                           self.to_mono)
            # load audio files from neighbour
            neighbour_path = os.path.join(self.dataset_path, neighbour.file_name)
            neighbour_audio = AudioUtils.load_audio_from_file(neighbour_path, self.sample_rate, self.stereo_channels,
                                                              self.to_mono)
            # load audio files from opposite
            opposite_path = os.path.join(self.dataset_path, opposite.file_name)
            opposite_audio = AudioUtils.load_audio_from_file(opposite_path, self.sample_rate, self.stereo_channels,
                                                             self.to_mono)

            # make sure audios have the same size
            audio_length = self.sample_size * self.sample_rate
            anchor_audio = anchor_audio[:audio_length]
            neighbour_audio = neighbour_audio[:audio_length]
            opposite_audio = opposite_audio[:audio_length]

            # create tuple of triplet labels
            triplet_labels = np.stack((anchor.label, neighbour.label, opposite.label), axis=0)
            if index % 1000 == 0 and self.log:
                self.logger.debug("Triplet labels, index: {0}, a: {1}, n: {2}, o: {3}".format(index,
                                                                                              anchor.label,
                                                                                              neighbour.label,
                                                                                              opposite.label))

            yield anchor_audio, neighbour_audio, opposite_audio, triplet_labels

    def get_dataset(self, feature_extractor: Union[Extractor, None], shuffle: bool = True, calc_dist: bool = False):
        # why not sample rate * sample_size?
        if self.to_mono:
            audio_shape = [self.sample_rate]
        else:
            audio_shape = [self.sample_rate, self.stereo_channels]

        dataset = tf.data.Dataset.from_generator(self.generate_samples,
                                                 args=[calc_dist],
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
        dataset = dataset.prefetch(self.prefetch_batches).repeat()

        return dataset

    def get_test_dataset(self, extractor):
        test_set = self.audio_info_df.get_test_set(self.stereo_channels, self.to_mono)
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
