import logging
import re
import warnings
from typing import Union, Tuple

import librosa
import numpy as np
import tensorflow as tf

from src.feature_extractor.base_extractor import BaseExtractor
from src.input_pipeline.base_dataset import BaseDataset, DatasetType
from src.utils.params import Params
from src.utils.utils_audio import AudioUtils


class TripletsInputPipeline:
    """Input pipeline to generate triplets."""

    def __init__(self, params: Params, dataset: BaseDataset, log: bool = False):
        """
        Initialises the audio pipeline.
        :param params: parameters of the current experiment.
        :param log: if the pipeline should log details about the data.
        """
        # ignore warnings, such as the librosa warnings
        warnings.filterwarnings('ignore')

        self.params = params

        self.dataset = dataset
        self.dataset_type = DatasetType.TRAIN

        self.log = log

        self.logger = logging.getLogger(self.__class__.__name__)

    def reinitialise(self):
        self.logger.info("Reinitialising the input pipeline")
        self.dataset.initialise()

    def __generate_samples(self, gen_name: str, trim: bool, return_labels: bool) -> Tuple[np.ndarray, np.ndarray,
                                                                                          np.ndarray, np.ndarray]:

        gen_name = gen_name.decode("utf-8")
        gen_index = int(re.findall('[0-9]+', gen_name)[0])

        self.dataset.current_index = gen_index
        for anchor in self.dataset:
            current_index = self.dataset.current_index - 1
            if self.log:
                self.logger.debug("{0}, {1}, index:{2}".format(gen_name, gen_index, current_index))

            # fill the opposite sample buffer
            opposite_audios = self.dataset.fill_opposite_selection(current_index)

            # load audio files from anchor
            anchor = self.dataset.df.iloc[current_index]
            if self.params.dataset == "MusicDataset":
                anchor_audio, _ = librosa.load(anchor.file_name, self.params.sample_rate)
                anchor_audio, _ = librosa.effects.trim(anchor_audio)
            else:
                anchor_audio = AudioUtils.load_audio_from_file(anchor.file_name, self.params.sample_rate,
                                                               self.params.sample_size,
                                                               self.params.stereo_channels,
                                                               self.params.to_mono)

            anchor_audio_length = int(len(anchor_audio) / self.params.sample_rate)

            try:
                triplets = self.dataset.get_triplets(current_index, anchor_audio_length, trim=trim,
                                                     opposite_choices=opposite_audios)
            except ValueError as err:
                self.logger.debug("Error during triplet creation: {}".format(err))
                continue

            for triplet in triplets:
                assert len(triplet) == 3, "Wrong shape of triplets."

                anchor_seg, neighbour_seg, opposite_seg = triplet

                # load audio files from neighbour
                opposite_entry = opposite_audios[opposite_seg[0]]
                opposite_audio = opposite_entry[0]

                # cut the tiles out of the audio files
                anchor_audio_seg = self.dataset.split_audio_in_segment(anchor_audio, anchor_seg[1])
                neighbour_audio_seg = self.dataset.split_audio_in_segment(anchor_audio, neighbour_seg[1])
                opposite_audio_seg = self.dataset.split_audio_in_segment(opposite_audio, opposite_seg[1])

                if self.dataset_type == DatasetType.EVAL or return_labels:
                    labels = [anchor.label, anchor.label, opposite_entry[1]]
                else:
                    labels = [-1, -1, -1]
                labels = np.asarray(labels)

                if current_index % 1000 == 0 and current_index is not 0 and self.log:
                    self.logger.debug("{0} yields sound segments {1}, a: {2}, n: {3}, o: {4}".format(gen_name,
                                                                                                     current_index,
                                                                                                     anchor_seg,
                                                                                                     neighbour_seg,
                                                                                                     opposite_seg))

                yield anchor_audio_seg, neighbour_audio_seg, opposite_audio_seg, labels

    def get_dataset(self, feature_extractor: Union[BaseExtractor, None], dataset_type: DatasetType = DatasetType.TRAIN,
                    shuffle: bool = True, trim: bool = True, return_labels: bool = False):

        self.dataset_type = dataset_type
        self.dataset.change_dataset_type(dataset_type)

        if self.params.to_mono:
            audio_shape = [self.params.sample_tile_size * self.params.sample_rate]
        else:
            audio_shape = [self.params.sample_tile_size * self.params.sample_rate, self.params.stereo_channels]

        gen_arr = ["Gen_{}".format(x) for x in range(self.params.gen_count)]
        dataset = tf.data.Dataset.from_tensor_slices(gen_arr)
        dataset = dataset.interleave(lambda gen_name: tf.data.Dataset.from_generator(self.__generate_samples,
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
                                     cycle_length=self.params.gen_count,
                                     block_length=1,
                                     num_parallel_calls=self.params.num_parallel_calls)

        # extract features from dataset
        if feature_extractor is not None:
            dataset = dataset.map(lambda a, n, o, labels: (
                feature_extractor.extract(a),
                feature_extractor.extract(n),
                feature_extractor.extract(o),
                labels), num_parallel_calls=self.params.num_parallel_calls)

        if shuffle:
            # buffer size defines from how much elements are in the buffer, from which then will get shuffled
            dataset = dataset.shuffle(buffer_size=self.params.random_selection_buffer_size)

        dataset = dataset.batch(self.params.batch_size)
        dataset = dataset.prefetch(self.params.prefetch_batches)

        return dataset
