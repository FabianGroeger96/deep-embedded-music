import logging
import math
import pathlib
import random
from typing import Union, Tuple

import librosa
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.input_pipeline.base_dataset import BaseDataset
from src.input_pipeline.dataset_factory import DatasetFactory
from src.utils.utils import Utils


@DatasetFactory.register("DJ")
class DJDataset(BaseDataset):
    AUDIO_FILES_DIR = "audio"
    INFO_FILES_DIR = "evaluation_setup"
    LABELS = ["DeepHouse", "Electronica_Downtempo", "IndieDance", "MelodicHouseAndTechno",
              "Techno_PeakTime_Driving_Hard", "Techno_Raw_Deep_Hypnotic", "Trance"]

    def __init__(self, dataset_path: Union[str, pathlib.Path], sample_rate: int,
                 train_test_split_distribution: int = 0.05, log: bool = False):

        super().__init__()

        self.dataset_path = Utils.check_if_path_exists(dataset_path)
        self.sample_rate = sample_rate
        self.train_test_split_distribution = train_test_split_distribution
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
        self.df_train, self.df_test = train_test_split(self.df, test_size=self.train_test_split_distribution)

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

        data = {'name': audio_names, "path": audio_paths, 'label': audio_labels}
        data = pd.DataFrame(data)
        data["label"] = data["label"].apply(self.LABELS.index)

        return data

    def get_triplets(self, anchor_id, calc_dist: bool = False, trim: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        try:
            anchor = self.df.iloc[anchor_id]
            neighbour, neighbour_dist = self.get_neighbour(anchor_id, calc_dist=calc_dist)
            opposite, opposite_dist = self.get_opposite(anchor_id, calc_dist=calc_dist)

            if calc_dist:
                self.check_if_easy_or_hard_triplet(neighbour_dist, opposite_dist)

            audio_anchor, _ = librosa.load(anchor["path"], self.sample_rate)
            audio_neighbour, _ = librosa.load(neighbour["path"], self.sample_rate)
            audio_opposite, _ = librosa.load(opposite["path"], self.sample_rate)

            if trim:
                # remove leading and trailing silence
                audio_anchor, _ = librosa.effects.trim(audio_anchor)
                audio_neighbour, _ = librosa.effects.trim(audio_neighbour)
                audio_opposite, _ = librosa.effects.trim(audio_opposite)

            # get audio length in seconds
            audio_anchor_length = librosa.get_duration(audio_anchor, self.sample_rate)
            audio_neighbour_length = librosa.get_duration(audio_neighbour, self.sample_rate)
            audio_opposite_length = librosa.get_duration(audio_opposite, self.sample_rate)

            # get the possible count of segments
            anchor_segments_count = math.floor(audio_anchor_length / 10)
            neighbour_segments_count = math.floor(audio_neighbour_length / 10)
            opposite_segments_count = math.floor(audio_opposite_length / 10)

            triplets = []
            labels = []

            for seg_id in range(anchor_segments_count):
                anchor_seg = audio_anchor[seg_id * 10 * self.sample_rate: (seg_id + 1) * 10 * self.sample_rate]

                neighbour_seg_id = random.randint(0, neighbour_segments_count - 1)
                neighbour_seg = self.split_audio_in_segment(audio_neighbour, neighbour_seg_id, 10)

                opposite_seg_id = random.randint(0, opposite_segments_count - 1)
                opposite_seg = self.split_audio_in_segment(audio_opposite, opposite_seg_id, 10)

                triplets.append([anchor_seg, neighbour_seg, opposite_seg])
                labels.append([anchor.label, neighbour.label, opposite.label])

            return np.asarray(triplets), np.asarray(labels)

        except ValueError as err:
            self.logger.debug("Error during triplet computation: {}".format(err))

    def get_neighbour(self, anchor_id, calc_dist: bool = False):
        anchor = self.df.iloc[anchor_id]

        filtered_items = self.df
        filtered_items = filtered_items[filtered_items.label == anchor.label]
        filtered_items = filtered_items[filtered_items["name"] != anchor["name"]]

        if len(filtered_items) > 0:
            if self.log:
                self.logger.debug("Selecting neighbour randomly from {} samples".format(len(filtered_items)))
        else:
            raise ValueError("No valid neighbour found")

        neighbour = filtered_items.sample().iloc[0]

        return neighbour, None

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

        return opposite, None

    def split_audio_in_segment(self, audio, segment_id, sample_size):
        segment = audio[segment_id * sample_size * self.sample_rate: (segment_id + 1) * sample_size * self.sample_rate]

        return segment
