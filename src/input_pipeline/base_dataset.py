import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple

import librosa
import numpy as np
from dtw import dtw
from numpy.linalg import norm


class DatasetType(Enum):
    TRAIN = 0
    EVAL = 1
    TEST = 2


class BaseDataset(ABC):
    EXPERIMENT_FOLDER = None
    LABELS = []

    def __init__(self):
        # defines the current index of the iterator
        self.sample_rate = None
        self.logger = None

        self.df = None
        self.df_train = None
        self.df_eval = None
        self.df_test = None

        self.opposite_indices = None
        self.opposite_audios = []

        self.current_index = 0
        self.log = None

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def load_data_frame(self):
        pass

    @abstractmethod
    def get_triplets(self, audio_id, audio_length, trim: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def get_neighbour(self, audio_id, anchor_sample_id: id, audio_length: int):
        pass

    @abstractmethod
    def get_opposite(self, audio_id, anchor_sample_id: id, audio_length: int):
        pass

    @abstractmethod
    def initialise(self):
        pass

    @abstractmethod
    def fill_opposite_selection(self, index):
        pass

    def change_dataset_type(self, dataset_type: DatasetType):
        self.current_index = 0
        if dataset_type == DatasetType.TRAIN:
            self.df = self.df_train
        elif dataset_type == DatasetType.EVAL:
            self.df = self.df_eval
        elif dataset_type == DatasetType.TRAIN:
            self.df = self.df_train

    def print_dataset_info(self):
        if self.log:
            self.logger.debug(self.df_train.head())

        self.count_classes()
        self.logger.info("Total audio samples: {}".format(self.df_train["file_name"].count()))

    def count_classes(self):
        label_counts = self.df_train["label"].value_counts()
        for i, label in enumerate(self.LABELS):
            if i < len(label_counts):
                self.logger.info("Audio samples in {0}: {1}".format(label, label_counts[i]))

    def check_if_easy_or_hard_triplet(self, neighbour_dist, opposite_dist):
        # distance to differ between hard / easy triplet
        if neighbour_dist > opposite_dist:
            self.logger.debug("Dist opposite smaller than dist neighbour --> hard triplet")
        else:
            self.logger.debug("Dist opposite bigger than dist neighbour --> easy triplet")

    def compare_audio(self, audio_1, audio_2):
        # compute MFCC from audio1
        audio_anchor, _ = librosa.load(os.path.join(self.dataset_path, audio_1.file_name), sr=self.sample_rate)
        mfcc_anchor = librosa.feature.mfcc(audio_anchor, self.sample_rate)
        # compute MFCC from audio2
        audio_neigh, _ = librosa.load(os.path.join(self.dataset_path, audio_2.file_name), sr=self.sample_rate)
        mfcc_neigh = librosa.feature.mfcc(audio_neigh, self.sample_rate)
        # compute distance between mfccs with dynamic-time-wrapping (dtw)
        dist, cost, acc_cost, path = dtw(mfcc_anchor.T, mfcc_neigh.T, dist=lambda x, y: norm(x - y, ord=1))

        return dist
