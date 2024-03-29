import logging
import pathlib

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.dataset.base_dataset import BaseDataset
from src.dataset.dataset_factory import DatasetFactory
from src.utils.params import Params
from src.utils.utils import Utils


@DatasetFactory.register("MusicDataset")
class MusicDataset(BaseDataset):
    """ Concrete implementation of the music / dj dataset, which consists out of songs of different genres. """

    EXPERIMENT_FOLDER = "DJ"
    LABELS = ["DeepHouse", "Electronica_Downtempo", "IndieDance", "MelodicHouseAndTechno",
              "Techno_PeakTime_Driving_Hard", "Techno_Raw_Deep_Hypnotic", "Trance"]

    def __init__(self, params: Params, log: bool = False):
        """
        Initialises the dataset.

        :param params: the global hyperparameters for initialising the dataset.
        :param log: if the dataset should provide a more detailed log.
        """
        super().__init__(params=params, log=log)

        self.dataset_path = Utils.check_if_path_exists(params.music_dataset_path)

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
        Splits the dataframes into training, evaluation and test dataset.
        `df_train`: contains the training set.
        `df_eval`: contains the evaluation set.
        `df_text`: contains the test set.
        """
        # set current index to start
        self.current_index = 0

        # load the data frame
        self.load_data_frame()

        # shuffle dataset
        self.df = self.df.sample(frac=1).reset_index(drop=True)

        # split dataset into train and test, test will be used for visualising
        self.df_train, self.df_eval = train_test_split(self.df, test_size=self.params.train_test_split)
        self.df_eval, self.df_test = train_test_split(self.df_eval, test_size=self.params.train_test_split)

    def load_data_frame(self):
        """
        Loads the dataframe from the audio files (.wav) located in the dataset path.
        """
        audio_names = []
        audio_paths = []
        audio_labels = []

        files = Utils.get_files_in_path(self.dataset_path, ".wav")
        for file in files:
            file_path = pathlib.Path(file)
            file_name = file_path.name
            file_label = file_path.parent.name

            audio_names.append(file_name)
            audio_paths.append(file_path)
            audio_labels.append(file_label)

        self.df = {'name': audio_names, "file_name": audio_paths, 'label': audio_labels}
        self.df = pd.DataFrame(self.df)
        self.df["label"] = self.df["label"].apply(self.LABELS.index)

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
            opposite_audio, _ = librosa.load(opposite_df.file_name, self.params.sample_rate)
            opposite_audio, _ = librosa.effects.trim(opposite_audio)
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
            for anchor_id in range(0, anchor_length - self.params.sample_tile_size, self.params.sample_tile_size):
                a_seg = [anchor_id, anchor_id]
                n_seg = self.get_neighbour(anchor_id, anchor_segment_id=anchor_id, anchor_length=anchor_length)
                o_seg = self.get_opposite(anchor_id, anchor_segment_id=anchor_id, anchor_length=anchor_length,
                                          opposite_choices=opposite_choices)

                triplets.append([a_seg, n_seg, o_seg])

            return np.asarray(triplets)

        except ValueError as err:
            self.logger.debug("Error during triplet computation: {}".format(err))
            raise ValueError("Error during triplet computation")

    def get_neighbour(self, anchor_id: int, anchor_segment_id: id, anchor_length: int):
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
        sample_possible = np.arange(0, anchor_length - self.params.sample_tile_size, self.params.sample_tile_size)

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
        neighbour_id = np.random.choice(sample_possible, 1)[0]

        return [anchor_id, neighbour_id]

    def get_opposite(self, anchor_id, anchor_segment_id: id, anchor_length: int, opposite_choices):
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
        opposite = np.random.choice(opposite_possible, size=1)[0]

        opposite_audio = opposite_choices[opposite][0]
        opposite_audio_length = int(len(opposite_audio) / self.params.sample_rate)

        # crate array of possible sample positions
        sample_possible = np.arange(0, opposite_audio_length - self.params.sample_tile_size,
                                    self.params.sample_tile_size)

        # random choose neighbour in possible samples
        opposite_id = np.random.choice(sample_possible, size=1)[0]

        return [opposite, opposite_id]
