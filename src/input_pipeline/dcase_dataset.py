import logging
import os
import random
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.input_pipeline.base_dataset import BaseDataset
from src.input_pipeline.dataset_factory import DatasetFactory
from src.utils.params import Params
from src.utils.utils import Utils
from src.utils.utils_audio import AudioUtils


@DatasetFactory.register("DCASEDataset")
class DCASEDataset(BaseDataset):
    EXPERIMENT_FOLDER = "DCASE"
    INFO_FILES_DIR = "evaluation_setup"
    LABELS = ["absence", "cooking", "dishwashing", "eating", "other", "social_activity", "vacuum_cleaner",
              "watching_tv", "working"]

    def __init__(self, params: Params, log: bool = False):

        super().__init__()

        self.params = params

        self.dataset_path = Utils.check_if_path_exists(params.dcase_dataset_path)
        self.fold = params.dcase_dataset_fold

        self.sample_rate = params.sample_rate
        self.sample_size = params.sample_size

        self.sample_tile_size = params.sample_tile_size
        self.sample_tile_neighbourhood = params.sample_tile_neighbourhood

        self.stereo_channels = params.stereo_channels
        self.to_mono = params.to_mono

        self.train_test_split = params.train_test_split
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
        self.df_train, self.df_test = train_test_split(self.df, test_size=self.train_test_split)

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

        return eval_df

    def load_train_data_frame(self):
        # define name and path of the info file
        train_file_name = "fold{0}_train.txt".format(self.fold)
        train_file_path = os.path.join(self.dataset_path, self.INFO_FILES_DIR, train_file_name)
        # read the eval file, which is tab separated
        train_df = pd.read_csv(train_file_path, sep="\t", names=["file_name", "label", "session"])
        # convert the activity labels into integers
        train_df["label"] = train_df["label"].apply(self.LABELS.index)

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

    def get_triplets(self, audio_id, trim: bool = True) -> np.ndarray:
        try:
            triplets = []
            for anchor_id in range(0, self.sample_size, self.sample_tile_size):
                a_seg = [audio_id, anchor_id]
                n_seg = self.get_neighbour(audio_id, anchor_sample_id=anchor_id, audio_length=self.sample_size)
                o_seg = self.get_opposite(audio_id, anchor_sample_id=anchor_id, audio_length=self.sample_size)

                triplets.append([a_seg, n_seg, o_seg])

            #
            # if calc_dist:
            #     self.check_if_easy_or_hard_triplet(neighbour_dist, opposite_dist)

            return np.asarray(triplets)

        except ValueError as err:
            self.logger.debug("Error during triplet computation: {}".format(err))
            raise ValueError("Error during triplet computation")

    def get_neighbour(self, audio_id: int, anchor_sample_id: id, audio_length: int):
        # crate array of possible sample positions
        sample_possible = np.arange(0, audio_length, self.sample_tile_size)

        # delete the current anchors id
        sample_possible = sample_possible[sample_possible != anchor_sample_id]

        # delete the sample ids which are not in range of the neighbourhood
        sample_possible = sample_possible[(sample_possible <= anchor_sample_id + self.sample_tile_neighbourhood) & (
                sample_possible >= anchor_sample_id - self.sample_tile_neighbourhood)]

        if len(sample_possible) > 0:
            if self.log:
                self.logger.debug("Selecting neighbour randomly from {} samples".format(len(sample_possible)))
        else:
            raise ValueError("No valid neighbour found")

        # random choose neighbour in possible samples
        neighbour_id = np.random.choice(sample_possible, 1)[0]

        return [audio_id, neighbour_id]

    def get_opposite(self, audio_id, anchor_sample_id: id, audio_length: int):
        # crate array of possible sample positions
        opposite_possible = np.arange(0, len(self.df), 1)
        opposite_possible = opposite_possible[opposite_possible != audio_id]

        if len(opposite_possible) > 0:
            if self.log:
                self.logger.debug("Selecting opposite randomly from {} samples".format(len(opposite_possible)))
        else:
            raise ValueError("No valid opposite found")

        opposite = random.choice(opposite_possible)

        # crate array of possible sample positions
        sample_possible = np.arange(0, audio_length, self.sample_tile_size)

        # random choose neighbour in possible samples
        opposite_id = np.random.choice(sample_possible, 1)[0]

        return [opposite, opposite_id]
