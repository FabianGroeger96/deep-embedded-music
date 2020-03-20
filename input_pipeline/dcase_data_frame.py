import logging
import math
import os
import pathlib
import re
from typing import Union

import librosa
import pandas as pd
from dtw import dtw
from numpy.linalg import norm

from utils.utils import Utils


class DCASEDataFrame:
    AUDIO_FILES_DIR = "audio"
    INFO_FILES_DIR = "evaluation_setup"
    LABELS = ["absence", "cooking", "dishwashing", "eating", "other", "social_activity", "vacuum_cleaner",
              "watching_tv", "working"]

    def __init__(self,
                 dataset_path: Union[str, pathlib.Path],
                 fold: int,
                 sample_rate: int):

        self.dataset_path = Utils.check_if_path_exists(dataset_path)
        self.fold = fold
        self.sample_rate = sample_rate

        # defines the current index of the iterator
        self.current_index = 0

        self.data_frame = self.load_data_frame()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(self.data_frame.head())

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self.current_index >= len(self.data_frame):
                raise StopIteration
            else:
                audio_entry = self.data_frame.iloc[self.current_index]

                if self.current_index % 1000 == 0:
                    self.logger.debug(
                        "Entry {0}: audio file: {1}, label: {2}, session: {3}, node id: {4}, segment: {5}".format(
                            self.current_index,
                            audio_entry.file_name,
                            audio_entry.label,
                            audio_entry.session,
                            audio_entry.node_id,
                            audio_entry.segment))

                self.current_index += 1

                return audio_entry

    @staticmethod
    def extract_info_from_filename(file_name):
        audio_name_split = file_name.split("/")
        assert len(audio_name_split) > 1, "Wrong audio file path"
        audio_name_split = audio_name_split[-1].split("_")

        assert len(audio_name_split) == 3, "Wrong audio file name"
        audio_node_id = re.findall(r"\d+", audio_name_split[0])[0]
        audio_session = re.findall(r"\d+", audio_name_split[1])[0]
        audio_segment = re.findall(r"\d+", audio_name_split[2])[0]

        return audio_node_id, audio_session, audio_segment

    def load_data_frame(self):
        train_df = self.load_train_data_frame()
        eval_df = self.load_eval_data_frame()

        return pd.concat([train_df, eval_df], axis=0, ignore_index=True)

    def load_eval_data_frame(self):
        # define name and path of the info file
        eval_file_name = "fold{0}_evaluate.txt".format(self.fold)
        eval_file_path = os.path.join(self.dataset_path, self.INFO_FILES_DIR, eval_file_name)
        # read the train file, which is tab separated
        eval_df = pd.read_csv(eval_file_path, sep="\t", names=["file_name", "label"])
        # convert the activity labels into integers
        eval_df["label"] = eval_df["label"].apply(self.LABELS.index)
        # create session column, because the file doesn't contains this info
        eval_df["session"] = ""
        # create two new columns: node_id and segment
        eval_df["node_id"] = ""
        eval_df["segment"] = ""
        # extract node_id, session and segment from sound file name
        eval_df["node_id"], eval_df["session"], eval_df["segment"] = zip(*eval_df.apply(
            lambda row: DCASEDataFrame.extract_info_from_filename(row["file_name"]), axis=1))
        return eval_df

    def load_train_data_frame(self):
        # define name and path of the info file
        train_file_name = "fold{0}_train.txt".format(self.fold)
        train_file_path = os.path.join(self.dataset_path, self.INFO_FILES_DIR, train_file_name)
        # read the eval file, which is tab separated
        train_df = pd.read_csv(train_file_path, sep="\t", names=["file_name", "label", "session"])
        # convert the activity labels into integers
        train_df["label"] = train_df["label"].apply(self.LABELS.index)
        # create two new columns: node_id and segment
        train_df["node_id"] = ""
        train_df["segment"] = ""
        # extract node_id, session and segment from sound file name
        train_df["node_id"], train_df["session"], train_df["segment"] = zip(*train_df.apply(
            lambda row: DCASEDataFrame.extract_info_from_filename(row["file_name"]), axis=1))
        return train_df

    def get_neighbour(self, anchor_id, calc_dist: bool = False, log_detail: bool = False):
        anchor = self.data_frame.iloc[anchor_id]

        filtered_items = self.data_frame
        filtered_items = filtered_items[filtered_items.label == anchor.label]
        filtered_items = filtered_items[filtered_items.session != anchor.session]
        filtered_items = filtered_items[filtered_items.node_id != anchor.node_id]

        if len(filtered_items) > 0:
            if log_detail:
                self.logger.debug("Selecting neighbour randomly from {} samples".format(len(filtered_items)))
        else:
            raise ValueError("No valid neighbour found")

        neighbour = filtered_items.sample().iloc[0]

        if calc_dist:
            dist = self.compare_audio(anchor, neighbour)
            if log_detail:
                self.logger.debug("Normalized distance between anchor and neighbour: {}".format(math.ceil(dist)))
        else:
            dist = None

        return neighbour, dist

    def get_opposite(self, anchor_id, calc_dist: bool = False, log_detail: bool = False):
        anchor = self.data_frame.iloc[anchor_id]

        filtered_items = self.data_frame
        filtered_items = filtered_items[filtered_items.label != anchor.label]

        if len(filtered_items) > 0:
            if log_detail:
                self.logger.debug("Selecting opposite randomly from {} samples".format(len(filtered_items)))
        else:
            raise ValueError("No valid opposite found")

        opposite = filtered_items.sample().iloc[0]

        if calc_dist:
            dist = self.compare_audio(anchor, opposite)
            if log_detail:
                self.logger.debug("Normalized distance between anchor and opposite: {}".format(math.ceil(dist)))
        else:
            dist = None

        return opposite, dist

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
