import logging
import os
from typing import Union, Tuple

import numpy as np
import tensorflow as tf

from src.feature_extractor.base_extractor import BaseExtractor
from src.input_pipeline.base_dataset import BaseDataset
from src.utils.params import Params
from src.utils.utils import Utils
from src.utils.utils_audio import AudioUtils


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

        self.sample_tile_size = params.sample_tile_size
        self.sample_tile_neighbourhood = params.sample_tile_neighbourhood

        self.stereo_channels = params.stereo_channels
        self.to_mono = params.to_mono

        self.batch_size = params.batch_size
        self.prefetch_batches = tf.data.experimental.AUTOTUNE  # tf.data.experimental.AUTOTUNE / params.prefetch_batches
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

    def generate_samples(self, gen_name: str, trim: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gen_name = gen_name.decode("utf-8")
        gen_index = int(gen_name[-1])
        start = int((self.batch_size / 4) * gen_index)
        end = int((self.batch_size / 4) * (gen_index + 1))

        self.dataset.current_index = start
        print_index = start
        for index, anchor in enumerate(self.dataset):

            try:
                triplets = self.dataset.get_triplets(index, trim=trim)
            except ValueError as err:
                self.logger.debug("Error during triplet creation: {}".format(err))
                continue

            # load audio files from anchor
            anchor = self.dataset.df.iloc[triplets[0][0][0]]
            anchor_audio = AudioUtils.load_audio_from_file(anchor.file_name, self.sample_rate, self.sample_size,
                                                           self.stereo_channels,
                                                           self.to_mono)

            for triplet in triplets:
                assert len(triplet) == 3, "Wrong shape of triplets."

                anchor_seg, neighbour_seg, opposite_seg = triplet

                # load audio files from neighbour
                opposite = self.dataset.df.iloc[opposite_seg[0]]
                opposite_audio = AudioUtils.load_audio_from_file(opposite.file_name, self.sample_rate, self.sample_size,
                                                                 self.stereo_channels,
                                                                 self.to_mono)

                # make sure audios have the same size
                audio_length = self.sample_size * self.sample_rate
                anchor_audio = anchor_audio[:audio_length]
                opposite_audio = opposite_audio[:audio_length]

                # cut the tiles out of the audio files
                anchor_audio_seg = anchor_audio[anchor_seg[1] * self.sample_rate:(anchor_seg[1] +
                                                                                  self.sample_tile_size) * self.sample_rate]
                neighbour_audio_seg = anchor_audio[neighbour_seg[1] * self.sample_rate:(neighbour_seg[1] +
                                                                                        self.sample_tile_size) * self.sample_rate]
                opposite_audio_seg = opposite_audio[opposite_seg[1] * self.sample_rate:(opposite_seg[1] +
                                                                                        self.sample_tile_size) * self.sample_rate]

                if print_index % 1000 == 0 and self.log:
                    self.logger.debug("{0} yields sound segments, a: {1}, n: {2}, o: {3}".format(gen_name,
                                                                                                 anchor_seg,
                                                                                                 neighbour_seg,
                                                                                                 opposite_seg))
                print_index += 1

                yield anchor_audio_seg, neighbour_audio_seg, opposite_audio_seg

    def get_dataset(self, feature_extractor: Union[BaseExtractor, None], shuffle: bool = True, trim: bool = True):

        if self.to_mono:
            audio_shape = [self.sample_tile_size * self.sample_rate]
        else:
            audio_shape = [self.sample_tile_size * self.sample_rate, self.stereo_channels]

        dataset = tf.data.Dataset.from_tensor_slices(['Gen_0', 'Gen_1', 'Gen_2', 'Gen_3'])
        dataset = dataset.interleave(lambda gen_name: tf.data.Dataset.from_generator(self.generate_samples,
                                                                                     args=[gen_name, trim],
                                                                                     output_shapes=(
                                                                                         tf.TensorShape(audio_shape),
                                                                                         tf.TensorShape(audio_shape),
                                                                                         tf.TensorShape(audio_shape)),
                                                                                     output_types=(
                                                                                         tf.float32, tf.float32,
                                                                                         tf.float32)),
                                     cycle_length=4,
                                     block_length=1,
                                     num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.cache()

        # extract features from dataset
        if feature_extractor is not None:
            dataset = dataset.map(lambda a, n, o: (
                feature_extractor.extract(a),
                feature_extractor.extract(n),
                feature_extractor.extract(o)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.cache()

        if shuffle:
            # buffer size defines from how much elements are in the buffer, from which then will get shuffled
            dataset = dataset.shuffle(buffer_size=self.random_selection_buffer_size)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
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
