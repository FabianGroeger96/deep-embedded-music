import logging
import os

import numpy as np
import pandas as pd

from src.dataset.base_dataset import BaseDataset
from src.dataset.dataset_factory import DatasetFactory
from src.utils.params import Params
from src.utils.utils import Utils
from src.utils.utils_audio import AudioUtils


@DatasetFactory.register("DCASEDataset")
class DCASEDataset(BaseDataset):
    """ Concrete implementation of the dataset from the DCASE Task 5 challenge, a noise detection dataset. """

    EXPERIMENT_FOLDER = "DCASE"
    INFO_FILES_DIR = "evaluation_setup"
    LABELS = ["absence", "cooking", "dishwashing", "eating", "other", "social_activity", "vacuum_cleaner",
              "watching_tv", "working"]

    def __init__(self, params: Params, log: bool = False):
        """
        Initialises the dataset.

        :param params: the global hyperparameters for initialising the dataset.
        :param log: if the dataset should provide a more detailed log.
        """
        super().__init__(params=params, log=log)

        self.dataset_path = Utils.check_if_path_exists(params.dcase_dataset_path)
        self.fold = params.dcase_dataset_fold
        self.eval_dataset_path = Utils.check_if_path_exists(params.dcase_eval_dataset_path)

        self.initialise()
        self.change_dataset_type(self.dataset_type)

        self.logger = logging.getLogger(self.__class__.__name__)

    def __iter__(self):
        """
        Returns an iterator for looping over the dataset.
        Since the class itself is an iterator, `self` is returned.
        """
        return self

    def __next__(self):
        """
        Specifies the retrieval of the next element in the dataset, by getting the entry of the datafame from the
        global current index. If the index is bigger than the length of the dataset, a `StopIteration` is raised.
        """
        while True:
            if self.current_index >= len(self.df):
                raise StopIteration
            else:
                audio_entry = self.df.iloc[self.current_index]
                self.current_index += 1

                return audio_entry

    def initialise(self):
        """
        Initialises the model.
        Loads the dataframes from the audio files and shuffles it.
        Resets the current index of the dataset to zero.
        """
        # set current index to start
        self.current_index = 0

        # load the data frame
        self.load_data_frame()

        # add the full dataset path to the filename
        self.df_train["file_name"] = str(self.dataset_path) + "/" + self.df_train["name"].astype(str)
        self.df_eval["file_name"] = str(self.dataset_path) + "/" + self.df_eval["name"].astype(str)
        self.df_test["file_name"] = str(self.eval_dataset_path) + "/" + self.df_test["name"].astype(str)

        self.df_train["name"] = self.df_train["name"].str.split("/", n=1, expand=True)[1]
        self.df_train["name"] = self.df_train["name"].str.split(".", n=1, expand=True)[0]

        self.df_eval["name"] = self.df_eval["name"].str.split("/", n=1, expand=True)[1]
        self.df_eval["name"] = self.df_eval["name"].str.split(".", n=1, expand=True)[0]

        self.df_test["name"] = self.df_test["name"].str.split("/", n=1, expand=True)[1]
        self.df_test["name"] = self.df_test["name"].str.split(".", n=1, expand=True)[0]

        # shuffle dataset
        self.df_train = self.df_train.sample(frac=1).reset_index(drop=True)
        self.df_eval = self.df_eval.sample(frac=1).reset_index(drop=True)
        self.df_test = self.df_test.sample(frac=1).reset_index(drop=True)

    def load_data_frame(self):
        """
        Loads the different dataframes.
        `df_train`: contains the training set.
        `df_eval`: contains the evaluation set.
        `df_test`: contains the test set (eval set from challenge)
        """
        self.__load_train_data_frame()
        self.__load_eval_data_frame()
        self.__load_test_data_frame()

    def __load_train_data_frame(self):
        """
        Loads the training dataset from the data, into a dataframe `df_train`.
        """
        # define name and path of the info file
        train_file_name = "fold{0}_train.txt".format(self.fold)
        train_file_path = os.path.join(self.dataset_path, self.INFO_FILES_DIR, train_file_name)
        # read the eval file, which is tab separated
        self.df_train = pd.read_csv(train_file_path, sep="\t", names=["name", "label", "session"])
        # convert the activity labels into integers
        self.df_train["label"] = self.df_train["label"].apply(self.LABELS.index)

    def __load_eval_data_frame(self):
        """
        Loads the evaluation dataset from the data, into a dataframe `df_eval`.
        """
        # define name and path of the info file
        eval_file_name = "fold{0}_evaluate.txt".format(self.fold)
        eval_file_path = os.path.join(self.dataset_path, self.INFO_FILES_DIR, eval_file_name)
        # read the train file, which is tab separated
        self.df_eval = pd.read_csv(eval_file_path, sep="\t", names=["name", "label"])
        # convert the activity labels into integers
        self.df_eval["label"] = self.df_eval["label"].apply(self.LABELS.index)

    def __load_test_data_frame(self):
        """
        Loads the test dataset from the data, into a dataframe `df_test`.
        This is the "evaluation" dataset from the challenge, which was used to compare the different models with
        each other.
        """
        # define name and path of the info file
        test_file_name = "evaluate.txt"
        test_file_path = os.path.join(self.eval_dataset_path, self.INFO_FILES_DIR, test_file_name)
        # read the train file, which is tab separated
        self.df_test = pd.read_csv(test_file_path, sep="\t", names=["name", "label", "session"])
        # convert the activity labels into integers
        self.df_test["label"] = self.df_test["label"].apply(self.LABELS.index)

    def fill_opposite_selection(self, anchor_id):
        """
        Fills up a list of opposite audio entries from a given anchor id `anchor_id`. The list will contain randomly
        selected loaded audios, which must not contain the anchors audio entry.

        :param anchor_id: the id of the audio entry in the dataset, which the anchor is belongs to.
        """
        opposite_possible = np.arange(0, len(self.df), 1)
        opposite_possible = opposite_possible[opposite_possible != anchor_id]

        opposite_indices = np.random.choice(opposite_possible, self.params.opposite_sample_buffer_size)
        opposite_audios = []
        for index in opposite_indices:
            opposite_df = self.df.iloc[index]
            opposite_audio = AudioUtils.load_audio_from_file(opposite_df.file_name, self.params.sample_rate,
                                                             self.params.sample_size,
                                                             self.params.stereo_channels,
                                                             self.params.to_mono)
            opposite_audios.append([opposite_audio, opposite_df.label, opposite_df["name"]])

        return opposite_audios

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
        try:
            triplets = []
            for anchor_id in range(0, anchor_length, self.params.sample_tile_size):
                a_seg = [anchor_id, anchor_id]
                n_seg = self.get_neighbour(anchor_id, anchor_segment_id=anchor_id, anchor_length=anchor_length)
                o_seg = self.get_opposite(anchor_id, anchor_segment_id=anchor_id, anchor_length=anchor_length,
                                          opposite_choices=opposite_choices)

                triplets.append([a_seg, n_seg, o_seg])

            return np.asarray(triplets)

        except ValueError as err:
            self.logger.debug("Error during triplet computation: {}".format(err))
            raise ValueError("Error during triplet computation")

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
        # crate array of possible sample positions
        sample_possible = np.arange(0, anchor_length, self.params.sample_tile_size)

        # delete the current anchors id
        sample_possible = sample_possible[sample_possible != anchor_segment_id]

        # delete the sample ids which are not in range of the neighbourhood
        sample_possible = sample_possible[
            (sample_possible <= anchor_segment_id + self.params.sample_tile_neighbourhood) & (
                    sample_possible >= anchor_segment_id - self.params.sample_tile_neighbourhood)]

        if len(sample_possible) > 0:
            if self.log:
                self.logger.debug("Selecting neighbour randomly from {} samples".format(len(sample_possible)))
        else:
            raise ValueError("No valid neighbour found")

        # random choose neighbour in possible samples
        neighbour_segment_id = np.random.choice(sample_possible, 1)[0]

        return [anchor_id, neighbour_segment_id]

    def get_opposite(self, anchor_id, anchor_segment_id: id, anchor_length: int, opposite_choices) -> []:
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
        # crate array of possible sample positions
        opposite_possible = np.arange(0, len(opposite_choices), 1)
        opposite_audio_id = np.random.choice(opposite_possible, size=1)[0]

        # crate array of possible sample positions
        sample_possible = np.arange(0, anchor_length, self.params.sample_tile_size)

        # random choose neighbour in possible samples
        opposite_segment_id = np.random.choice(sample_possible, size=1)[0]

        return [opposite_audio_id, opposite_segment_id]
