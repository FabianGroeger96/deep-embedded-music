import logging
import math
import pathlib
import random
from typing import Tuple

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.input_pipeline.base_dataset import BaseDataset
from src.input_pipeline.dataset_factory import DatasetFactory
from src.utils.params import Params
from src.utils.utils import Utils
from src.utils.utils_audio import AudioUtils


@DatasetFactory.register("MusicDataset")
class MusicDataset(BaseDataset):
    EXPERIMENT_FOLDER = "DJ"
    LABELS = ["DeepHouse", "Electronica_Downtempo", "IndieDance", "MelodicHouseAndTechno",
              "Techno_PeakTime_Driving_Hard", "Techno_Raw_Deep_Hypnotic", "Trance"]

    def __init__(self, params: Params, log: bool = False):

        super().__init__()

        self.params = params

        self.dataset_path = Utils.check_if_path_exists(params.music_dataset_path)

        self.sample_rate = params.sample_rate

        self.sample_tile_size = params.sample_tile_size
        self.sample_tile_neighbourhood = params.sample_tile_neighbourhood

        self.train_test_split = params.train_test_split
        self.log = log

        self.initialise()

        self.logger = logging.getLogger(self.__class__.__name__)
        if self.log:
            self.logger.debug(self.df.head())

        self.count_classes()
        self.logger.info("Total audio samples: {}".format(self.df["name"].count()))

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self.current_index >= len(self.df):
                raise StopIteration
            else:
                audio_entry = self.df.iloc[self.current_index]

                if self.current_index % 20 == 0 and self.log:
                    self.logger.debug(
                        "Entry {0}: audio file: {1}, audio path: {2}, label: {3}".format(
                            self.current_index,
                            audio_entry.name,
                            audio_entry.path,
                            audio_entry.label))

                self.current_index += 1

                return audio_entry

    def initialise(self):
        # load the data frame
        self.df = self.load_data_frame()
        # shuffle dataset
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        # split dataset into train and test, test will be used for visualising
        self.df_train, self.df_test = train_test_split(self.df, test_size=self.train_test_split)

    def load_data_frame(self):
        audio_names = []
        audio_paths = []
        audio_labels = []

        files = Utils.get_files_in_path(self.dataset_path, ".mp3")
        for file in files:
            file_path = pathlib.Path(file)
            file_name = file_path.name
            file_label = file_path.parent.name

            audio_names.append(file_name)
            audio_paths.append(file_path)
            audio_labels.append(file_label)

        data = {'name': audio_names, "file_name": audio_paths, 'label': audio_labels}
        data = pd.DataFrame(data)
        data["label"] = data["label"].apply(self.LABELS.index)

        return data

    def fill_opposite_selection(self, audio_id):
        opposite_possible = np.arange(0, len(self.df), 1)
        opposite_possible = opposite_possible[opposite_possible != audio_id]
        self.opposite_indices = np.random.choice(opposite_possible, 3)
        self.opposite_audios = []
        for index in self.opposite_indices:
            opposite_df = self.df.iloc[index]
            opposite_audio, _ = librosa.load(opposite_df.file_name, self.sample_rate)
            opposite_audio, _ = librosa.effects.trim(opposite_audio)
            self.opposite_audios.append(opposite_audio)

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
        sample_possible = np.arange(0, audio_length - self.sample_tile_size, self.sample_tile_size)

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
        opposite_possible = np.arange(0, len(self.opposite_indices), 1)
        opposite = random.choice(opposite_possible)

        opposite_audio = self.opposite_audios[opposite]
        opposite_audio_length = int(len(opposite_audio) / self.sample_rate)

        # crate array of possible sample positions
        sample_possible = np.arange(0, opposite_audio_length - self.sample_tile_size, self.sample_tile_size)

        # random choose neighbour in possible samples
        opposite_id = np.random.choice(sample_possible, 1)[0]

        return [opposite, opposite_id]

    def split_audio_in_segment(self, audio, segment_id, sample_size):
        segment = audio[segment_id * sample_size * self.sample_rate: (segment_id + 1) * sample_size * self.sample_rate]

        return segment
