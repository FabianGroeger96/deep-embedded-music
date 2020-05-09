import os
from abc import ABC, abstractmethod
from enum import Enum

import librosa
import numpy as np
import pandas as pd
from dtw import dtw
from numpy.linalg import norm

from src.utils.params import Params


class DatasetType(Enum):
    """
    Enumeration that specifies what dataset to use.

    TRAIN: used to train the model.
    EVAL: used to evaluate the model.
    TEST: used to test the model.
    """
    TRAIN = 0
    EVAL = 1
    TEST = 2
    FULL = 3


class BaseDataset(ABC):
    """ The abstract base dataset implementation. All datasets will inherit this base. """

    EXPERIMENT_FOLDER = None
    LABELS = []

    def __init__(self, params: Params, log: bool = False):
        """
        Initialises the dataset. Default dataset type is TRAIN.

        :param params: the global hyperparameters for initialising the dataset.
        :param log: if the dataset should provide a more detailed log.
        """
        self.params = params
        self.log = log

        # set the dataset type (e.g. train, eval, test)
        self.dataset_type = DatasetType.TRAIN

        self.logger = None

        self.df = None
        self.df_train = None
        self.df_eval = None
        self.df_test = None

        self.dataset_path = None

        # defines the current index of the iterator
        self.current_index = 0

        # set the random seed
        np.random.seed(params.random_seed)

    @abstractmethod
    def __iter__(self):
        """
        Returns an iterator for looping over the dataset.
        """
        pass

    @abstractmethod
    def __next__(self):
        """
        Specifies the retrieval of the next element of the dataset.
        """
        pass

    @abstractmethod
    def initialise(self):
        """
        Initialises the dataset.
        """
        pass

    @abstractmethod
    def load_data_frame(self):
        """
        Loads the dataframes with the data of the dataset.
        """
        pass

    @abstractmethod
    def fill_opposite_selection(self, anchor_id):
        """
        Fills up a list of opposite audio entries from a given anchor id `anchor_id`. The list will contain randomly
        selected loaded audios, which must not contain the anchors audio entry.

        :param anchor_id: the id of the audio entry in the dataset, which the anchor is belongs to.
        """
        pass

    @abstractmethod
    def get_triplets(self, anchor_id, anchor_length, opposite_choices) -> np.ndarray:
        """
        Calculates a triplet from a given anchor id.

        :param anchor_id: the id of the audio entry in the dataset, which the anchor is belongs to.
        :param anchor_length: the length of the audio file of the anchor.
        :param opposite_choices: a list of loaded audio files to choose the opposite segment from. this is mainly used
            to speed up the process of triplet selection in very large audio files.
        :return: triplets of audio segments from the given anchor audio,
            `[
                [[anchor_segment], [neighbour_segment], [opposite_segment]],
                [[anchor_segment], [neighbour_segment], [opposite_segment]]
            ]`
        """
        pass

    @abstractmethod
    def get_neighbour(self, anchor_id: int, anchor_segment_id: id, anchor_length: int) -> []:
        """
        Gets the neighbour segment from a given anchor id and anchor segment id.

        :param anchor_id: the id of the audio entry in the dataset, which the anchor is belongs to.
        :param anchor_segment_id: the id of the audio segment of the audio file, which will be used as anchor.
        :param anchor_length: the length of the anchor audio.
        :return: neighbour segment from a given anchor audio and anchor id.
            the segment consists of the audio the segment belongs to and the segment id.
            the audio id of the neighbour segment is the same as for the anchor audio.
            `[anchor_audio_id, neighbour_segment_id]`
        """
        pass

    @abstractmethod
    def get_opposite(self, anchor_id: int, anchor_segment_id: int, anchor_length: int, opposite_choices: []):
        """
        Gets the opposite segment from a given anchor id and anchor segment id.

        :param anchor_id: the id of the audio entry in the dataset, which the anchor is belongs to.
        :param anchor_segment_id: the id of the audio segment of the audio file, which will be used as anchor.
        :param anchor_length: the length of the anchor audio.
        :param opposite_choices: a list of loaded audio files to choose the opposite segment from. this is mainly used
            to speed up the process of selecting the opposite segment in very large audio files.
        :return: opposite segment from a given anchor audio and anchor id.
            the audio id of the opposite has to be different from the anchor audio id.
            `[opposite_audio_id, opposite_segment_id]`
        """
        pass

    def change_dataset_type(self, dataset_type: DatasetType):
        """
        Changes the current dataset type.

        :param dataset_type: the dataset type to change to.
        """
        self.current_index = 0
        if dataset_type == DatasetType.TRAIN:
            self.df = self.df_train
        elif dataset_type == DatasetType.EVAL:
            self.df = self.df_eval
        elif dataset_type == DatasetType.TEST:
            self.df = self.df_test
        elif dataset_type == DatasetType.FULL:
            self.df = pd.concat([self.df_train, self.df_eval, self.df_test])

        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def print_dataset_info(self):
        """
        Prints the info of the dataset such as the head of the dataframe, the count of total classes and the
        number of total audio samples in the dataset.
        """
        if self.log:
            self.logger.debug(self.df.head())

        self.__count_classes()
        self.logger.info("Total audio samples: {}".format(self.df["file_name"].count()))

    def __count_classes(self):
        """
        Counts the number of classes in the dataset and logs it.
        """
        label_counts = self.df["label"].value_counts()
        for i, label in enumerate(self.LABELS):
            if i < len(label_counts):
                self.logger.info("Audio samples in {0}: {1}".format(label, label_counts[i]))

    def split_audio_in_segment(self, audio, segment_id):
        """
        Cuts out a segment of a given audio segment, by giving the id of the segment.

        :param audio: the audio to cut the segment out.
        :param segment_id: the segment id to cut out of the audio.
        """
        segment = audio[segment_id * self.params.sample_rate:
                        (segment_id + self.params.sample_tile_size) * self.params.sample_rate]

        return segment

    def check_if_easy_or_hard_triplet(self, neighbour_dist, opposite_dist):
        """
        Checks if the distance between the neighbour and opposite distance classifies as hard or easy triplet.

        :param neighbour_dist: distance of the anchor to the neighbour.
        :param opposite_dist: distance of the anchor to the opposite.
        """
        # distance to differ between hard / easy triplet
        if neighbour_dist > opposite_dist:
            self.logger.debug("Dist opposite smaller than dist neighbour --> hard triplet")
        else:
            self.logger.debug("Dist opposite bigger than dist neighbour --> easy triplet")

    def compare_audio(self, audio_1, audio_2):
        """
        Compares two given audios by calculating their distances from each other, using dynamic-time-wrapping from
        the MFCC.

        :param audio_1: the first audio to compare.
        :param audio_2: the second audio to compare.
        """
        # compute MFCC from audio1
        audio_anchor, _ = librosa.load(os.path.join(self.dataset_path, audio_1.file_name),
                                       sr=self.params.sample_rate)
        mfcc_anchor = librosa.feature.mfcc(audio_anchor, self.params.sample_rate)
        # compute MFCC from audio2
        audio_neigh, _ = librosa.load(os.path.join(self.dataset_path, audio_2.file_name),
                                      sr=self.params.sample_rate)
        mfcc_neigh = librosa.feature.mfcc(audio_neigh, self.params.sample_rate)
        # compute distance between mfccs with dynamic-time-wrapping (dtw)
        dist, cost, acc_cost, path = dtw(mfcc_anchor.T, mfcc_neigh.T, dist=lambda x, y: norm(x - y, ord=1))

        return dist
