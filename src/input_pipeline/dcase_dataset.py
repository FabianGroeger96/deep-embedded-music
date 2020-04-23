import logging
import os
import random
import re

import numpy as np
import pandas as pd

from src.input_pipeline.base_dataset import BaseDataset, DatasetType
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

        self.dataset_type = DatasetType.TRAIN

        self.train_test_split = params.train_test_split
        self.log = log

        self.initialise()
        self.change_dataset_type(self.dataset_type)

        self.logger = logging.getLogger(self.__class__.__name__)

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
        # set current index to start
        self.current_index = 0

        # load the data frame
        self.load_data_frame()

        # add the full dataset path to the filename
        self.df_train["file_name"] = str(self.dataset_path) + "/" + self.df_train["file_name"].astype(str)
        self.df_eval["file_name"] = str(self.dataset_path) + "/" + self.df_eval["file_name"].astype(str)
        self.df_test["file_name"] = str(self.dataset_path) + "/" + self.df_test["file_name"].astype(str)

        # shuffle dataset
        self.df_train = self.df_train.sample(frac=1).reset_index(drop=True)
        self.df_eval = self.df_eval.sample(frac=1).reset_index(drop=True)
        self.df_test = self.df_test.sample(frac=1).reset_index(drop=True)

    def load_data_frame(self):
        self.df_train = self.load_train_data_frame()
        self.df_eval = self.load_eval_data_frame()
        self.df_test = self.load_test_data_frame()

    def load_train_data_frame(self):
        # define name and path of the info file
        train_file_name = "fold{0}_train.txt".format(self.fold)
        train_file_path = os.path.join(self.dataset_path, self.INFO_FILES_DIR, train_file_name)
        # read the eval file, which is tab separated
        train_df = pd.read_csv(train_file_path, sep="\t", names=["file_name", "label", "session"])
        # convert the activity labels into integers
        train_df["label"] = train_df["label"].apply(self.LABELS.index)

        return train_df

    def load_eval_data_frame(self):
        # define name and path of the info file
        eval_file_name = "fold{0}_evaluate.txt".format(self.fold)
        eval_file_path = os.path.join(self.dataset_path, self.INFO_FILES_DIR, eval_file_name)
        # read the train file, which is tab separated
        eval_df = pd.read_csv(eval_file_path, sep="\t", names=["file_name", "label"])
        # convert the activity labels into integers
        eval_df["label"] = eval_df["label"].apply(self.LABELS.index)

        return eval_df

    def load_test_data_frame(self):
        # define name and path of the info file
        test_file_name = "fold{0}_test.txt".format(self.fold)
        test_file_path = os.path.join(self.dataset_path, self.INFO_FILES_DIR, test_file_name)
        # read the train file, which is tab separated
        test_df = pd.read_csv(test_file_path, sep="\t", names=["file_name"])

        return test_df

    def fill_opposite_selection(self, audio_id):
        # TODO eval if needed
        pass

    def get_triplets(self, audio_id, audio_length, trim: bool = True) -> np.ndarray:
        try:
            triplets = []
            for anchor_id in range(0, audio_length, self.sample_tile_size):
                a_seg = [audio_id, anchor_id]
                n_seg = self.get_neighbour(audio_id, anchor_sample_id=anchor_id, audio_length=audio_length)
                o_seg = self.get_opposite(audio_id, anchor_sample_id=anchor_id, audio_length=audio_length)

                triplets.append([a_seg, n_seg, o_seg])

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
