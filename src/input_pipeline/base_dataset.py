import os
from abc import ABC, abstractmethod
from typing import Tuple

import librosa
import numpy as np
from dtw import dtw
from numpy.linalg import norm


class BaseDataset(ABC):
    EXPERIMENT_FOLDER = None
    LABELS = []

    def __init__(self):
        # defines the current index of the iterator
        self.sample_rate = None
        self.logger = None
        self.df = None
        self.current_index = 0

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
    def get_triplets(self, anchor_id, calc_dist: bool = False, trim: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def get_neighbour(self, anchor_id, calc_dist: bool = False):
        pass

    @abstractmethod
    def get_opposite(self, anchor_id, calc_dist: bool = False):
        pass

    @abstractmethod
    def initialise(self):
        pass

    def count_classes(self):
        label_counts = self.df["label"].value_counts()
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
