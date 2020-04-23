import logging
from typing import Union, Tuple

import librosa
import numpy as np
import tensorflow as tf

from src.feature_extractor.base_extractor import BaseExtractor
from src.input_pipeline.base_dataset import BaseDataset, DatasetType
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
        self.dataset_name = params.dataset

        self.fold = params.dcase_dataset_fold

        self.num_parallel_calls = params.num_parallel_calls
        self.gen_count = params.gen_count
        self.gen_index = 0

        self.sample_rate = params.sample_rate
        self.sample_size = params.sample_size

        self.sample_tile_size = params.sample_tile_size
        self.sample_tile_neighbourhood = params.sample_tile_neighbourhood

        self.stereo_channels = params.stereo_channels
        self.to_mono = params.to_mono

        self.batch_size = params.batch_size
        self.prefetch_batches = params.prefetch_batches
        self.random_selection_buffer_size = params.random_selection_buffer_size

        self.dataset_type = DatasetType.TRAIN

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
        self.gen_index = 0
        self.dataset.initialise()

    def generate_samples(self, gen_name: str, trim: bool, return_labels: bool) -> Tuple[np.ndarray, np.ndarray,
                                                                                        np.ndarray, np.ndarray]:

        gen_name = gen_name.decode("utf-8")

        self.dataset.current_index = self.gen_index
        for index, anchor in enumerate(self.dataset):
            self.dataset.current_index = self.gen_index
            if self.log and False:
                self.logger.debug("{0}, index:{1}".format(gen_name, self.gen_index))

            # load audio files from anchor
            anchor = self.dataset.df_train.iloc[index]
            if self.dataset_name == "MusicDataset":
                anchor_audio, _ = librosa.load(anchor.file_name, self.sample_rate)
                anchor_audio, _ = librosa.effects.trim(anchor_audio)
                self.dataset.fill_opposite_selection(index)
            else:
                anchor_audio = AudioUtils.load_audio_from_file(anchor.file_name, self.sample_rate, self.sample_size,
                                                               self.stereo_channels,
                                                               self.to_mono)

            anchor_audio_length = int(len(anchor_audio) / self.sample_rate)

            try:
                triplets = self.dataset.get_triplets(index, anchor_audio_length, trim=trim)
            except ValueError as err:
                self.logger.debug("Error during triplet creation: {}".format(err))
                continue

            for triplet in triplets:
                assert len(triplet) == 3, "Wrong shape of triplets."

                anchor_seg, neighbour_seg, opposite_seg = triplet

                # load audio files from neighbour
                if self.dataset_name == "MusicDataset":
                    opposite_audio = self.dataset.opposite_audios[opposite_seg[0]]
                else:
                    opposite = self.dataset.df_train.iloc[opposite_seg[0]]
                    opposite_audio = AudioUtils.load_audio_from_file(opposite.file_name, self.sample_rate, self.sample_size,
                                                                     self.stereo_channels,
                                                                     self.to_mono)

                # cut the tiles out of the audio files
                anchor_audio_seg = anchor_audio[anchor_seg[1] * self.sample_rate:(anchor_seg[1] +
                                                                                  self.sample_tile_size) * self.sample_rate]
                neighbour_audio_seg = anchor_audio[neighbour_seg[1] * self.sample_rate:(neighbour_seg[1] +
                                                                                        self.sample_tile_size) * self.sample_rate]
                opposite_audio_seg = opposite_audio[opposite_seg[1] * self.sample_rate:(opposite_seg[1] +
                                                                                        self.sample_tile_size) * self.sample_rate]

                if self.dataset_type == DatasetType.EVAL or return_labels:
                    labels = [anchor.label, anchor.label, opposite.label]
                else:
                    labels = [-1, -1, -1]
                labels = np.asarray(labels)

                if self.gen_index % 1000 == 0 and self.gen_index is not 0 and self.log:
                    self.logger.debug("{0} yields sound segments {1}, a: {2}, n: {3}, o: {4}".format(gen_name,
                                                                                                     self.dataset.current_index,
                                                                                                     anchor_seg,
                                                                                                     neighbour_seg,
                                                                                                     opposite_seg))

                yield anchor_audio_seg, neighbour_audio_seg, opposite_audio_seg, labels

            self.gen_index += 1

    def get_dataset(self, feature_extractor: Union[BaseExtractor, None], dataset_type: DatasetType = DatasetType.TRAIN,
                    shuffle: bool = True, trim: bool = True, return_labels: bool = False):

        self.dataset_type = dataset_type
        self.dataset.change_dataset_type(dataset_type)

        if self.to_mono:
            audio_shape = [self.sample_tile_size * self.sample_rate]
        else:
            audio_shape = [self.sample_tile_size * self.sample_rate, self.stereo_channels]

        gen_arr = ["Gen_{}".format(x) for x in range(self.gen_count)]
        dataset = tf.data.Dataset.from_tensor_slices(gen_arr)
        dataset = dataset.interleave(lambda gen_name: tf.data.Dataset.from_generator(self.generate_samples,
                                                                                     args=[gen_name, trim,
                                                                                           return_labels],
                                                                                     output_shapes=(
                                                                                         tf.TensorShape(audio_shape),
                                                                                         tf.TensorShape(audio_shape),
                                                                                         tf.TensorShape(audio_shape),
                                                                                         tf.TensorShape(3)),
                                                                                     output_types=(
                                                                                         tf.float32, tf.float32,
                                                                                         tf.float32, tf.float32)),
                                     cycle_length=self.gen_count,
                                     block_length=1,
                                     num_parallel_calls=self.num_parallel_calls)

        # extract features from dataset
        if feature_extractor is not None:
            dataset = dataset.map(lambda a, n, o, labels: (
                feature_extractor.extract(a),
                feature_extractor.extract(n),
                feature_extractor.extract(o),
                labels), num_parallel_calls=self.num_parallel_calls)

        if shuffle:
            # buffer size defines from how much elements are in the buffer, from which then will get shuffled
            dataset = dataset.shuffle(buffer_size=self.random_selection_buffer_size)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.prefetch_batches)

        return dataset
