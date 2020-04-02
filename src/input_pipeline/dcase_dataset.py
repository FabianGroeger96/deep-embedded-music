import logging
import math
import os
import pathlib
import re
from typing import Union, Tuple

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.input_pipeline.base_dataset import BaseDataset
from src.input_pipeline.dataset_factory import DatasetFactory
from src.utils.utils_audio import AudioUtils
from src.utils.utils import Utils


@DatasetFactory.register("DCASE")
class DCASEDataset(BaseDataset):
    AUDIO_FILES_DIR = "audio"
    INFO_FILES_DIR = "evaluation_setup"
    LABELS = ["absence", "cooking", "dishwashing", "eating", "other", "social_activity", "vacuum_cleaner",
              "watching_tv", "working"]

    def __init__(self, dataset_path: Union[str, pathlib.Path], fold: int, sample_rate: int,
                 sample_size: int, stereo_channels: int, to_mono: bool, train_test_split_distribution: int = 0.05,
                 log: bool = False):

        super().__init__()

        self.dataset_path = Utils.check_if_path_exists(dataset_path)
        self.fold = fold
        self.sample_rate = sample_rate
        self.sample_size = sample_size
        self.stereo_channels = stereo_channels
        self.to_mono = to_mono
        self.train_test_split_distribution = train_test_split_distribution
        self.log = log

        self.initialise()

        self.logger = logging.getLogger(self.__class__.__name__)
        if self.log:
            self.logger.debug(self.df.head())

        self.count_classes()
        self.logger.info("Total audio samples: {}".format(self.df["file_name"].count()))

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self.current_index >= len(self.df):
                raise StopIteration
            else:
                audio_entry = self.df.iloc[self.current_index]

                if self.current_index % 1000 == 0 and self.log:
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

    @staticmethod
    def extract_audio_from_filename(file_name, dataset_path, sample_rate, sample_size, stereo_channels, to_mono):
        audio_path = os.path.join(dataset_path, file_name)
        audio_data = AudioUtils.load_audio_from_file(audio_path, sample_rate, sample_size, stereo_channels, to_mono)

        return audio_data

    def initialise(self):
        # load the data frame
        self.df = self.load_data_frame()
        # shuffle dataset
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        # split dataset into train and test, test will be used for visualising
        self.df_train, self.df_test = train_test_split(self.df, test_size=self.train_test_split_distribution)

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
            lambda row: DCASEDataset.extract_info_from_filename(row["file_name"]), axis=1))
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
            lambda row: DCASEDataset.extract_info_from_filename(row["file_name"]), axis=1))
        return train_df

    def count_classes(self):
        label_counts = self.df["label"].value_counts()
        for i, label in enumerate(self.LABELS):
            if i < len(label_counts):
                self.logger.info("Audio samples in {0}: {1}".format(label, label_counts[i]))

    def get_test_set(self, stereo_channels, to_mono):
        self.df_test = self.df_test.drop(["node_id", "segment", "session"], axis=1)
        self.df_test["audio"] = ""

        self.df_test["audio"] = self.df_test.apply(lambda row: DCASEDataset.extract_audio_from_filename(
            row["file_name"], self.dataset_path, self.sample_rate, self.sample_size, stereo_channels, to_mono), axis=1)
        self.df_test = self.df_test.drop(["file_name"], axis=1)

        self.df_test = self.df_test[["audio", "label"]]

        return self.df_test

    def get_triplets(self, anchor_id, calc_dist: bool = False, trim: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        try:
            anchor = self.df.iloc[anchor_id]
            neighbour, neighbour_dist = self.get_neighbour(anchor_id, calc_dist=calc_dist)
            opposite, opposite_dist = self.get_opposite(anchor_id, calc_dist=calc_dist)

            if calc_dist:
                self.check_if_easy_or_hard_triplet(neighbour_dist, opposite_dist)

            # load audio files from anchor
            anchor_path = os.path.join(self.dataset_path, anchor.file_name)
            anchor_audio = AudioUtils.load_audio_from_file(anchor_path, self.sample_rate, self.sample_size,
                                                           self.stereo_channels,
                                                           self.to_mono)
            # load audio files from neighbour
            neighbour_path = os.path.join(self.dataset_path, neighbour.file_name)
            neighbour_audio = AudioUtils.load_audio_from_file(neighbour_path, self.sample_rate, self.sample_size,
                                                              self.stereo_channels, self.to_mono)
            # load audio files from opposite
            opposite_path = os.path.join(self.dataset_path, opposite.file_name)
            opposite_audio = AudioUtils.load_audio_from_file(opposite_path, self.sample_rate, self.sample_size,
                                                             self.stereo_channels, self.to_mono)

            triplets = [anchor_audio, neighbour_audio, opposite_audio]
            triplets = tf.stack(triplets, axis=0)
            triplets = tf.expand_dims(triplets, axis=0)

            labels = [anchor.label, neighbour.label, opposite.label]
            labels = np.expand_dims(labels, axis=0)

            return np.asarray(triplets), np.asarray(labels)

        except ValueError as err:
            self.logger.debug("Error during triplet computation: {}".format(err))

    def get_neighbour(self, anchor_id, calc_dist: bool = False):
        anchor = self.df.iloc[anchor_id]

        filtered_items = self.df
        filtered_items = filtered_items[filtered_items.label == anchor.label]
        filtered_items = filtered_items[filtered_items.session != anchor.session]
        filtered_items = filtered_items[filtered_items.node_id != anchor.node_id]

        if len(filtered_items) > 0:
            if self.log:
                self.logger.debug("Selecting neighbour randomly from {} samples".format(len(filtered_items)))
        else:
            raise ValueError("No valid neighbour found")

        neighbour = filtered_items.sample().iloc[0]

        if calc_dist:
            dist = self.compare_audio(anchor, neighbour)
            if self.log:
                self.logger.debug("Normalized distance between anchor and neighbour: {}".format(math.ceil(dist)))
        else:
            dist = None

        return neighbour, dist

    def get_opposite(self, anchor_id, calc_dist: bool = False):
        anchor = self.df.iloc[anchor_id]

        filtered_items = self.df
        filtered_items = filtered_items[filtered_items.label != anchor.label]

        if len(filtered_items) > 0:
            if self.log:
                self.logger.debug("Selecting opposite randomly from {} samples".format(len(filtered_items)))
        else:
            raise ValueError("No valid opposite found")

        opposite = filtered_items.sample().iloc[0]

        if calc_dist:
            dist = self.compare_audio(anchor, opposite)
            if self.log:
                self.logger.debug("Normalized distance between anchor and opposite: {}".format(math.ceil(dist)))
        else:
            dist = None

        return opposite, dist
