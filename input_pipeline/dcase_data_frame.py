import os
import pathlib
import re
from typing import Union

import pandas as pd

from utils.utils import Utils


class DCASEDataFrame:
    LABELS = ["absence", "cooking", "dishwashing", "eating", "other", "social_activity", "vacuum_cleaner",
              "watching_tv", "working"]

    def __init__(self,
                 audio_files_path: Union[str, pathlib.Path],
                 info_file_path: Union[str, pathlib.Path],
                 sample_rate: int):
        self.current_index = 0
        self.sample_rate = sample_rate

        self.audio_files_path = Utils.check_if_path_exists(audio_files_path)
        self.info_file_path = Utils.check_if_path_exists(info_file_path)

        self.data_frame = pd.read_csv(self.info_file_path, sep='\t', names=['sound_file',
                                                                            'activity_label',
                                                                            'session'])
        self.data_frame["node_id"] = ""
        self.data_frame["segment"] = ""
        self.data_frame["activity_label"] = self.data_frame["activity_label"].apply(self.LABELS.index)

        self.data_frame["node_id"], self.data_frame["session"], self.data_frame["segment"] = zip(*self.data_frame.apply(
            lambda row: DCASEDataFrame.extract_info_from_filename(row["sound_file"]), axis=1))

        print(self.data_frame.head())

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self.current_index > len(self.data_frame):
                raise StopIteration
            else:
                audio_info = self.data_frame.iloc[self.current_index]

                audio_file = audio_info.sound_file
                audio_label = audio_info.activity_label
                audio_session = audio_info.session
                audio_node_id = audio_info.node_id
                audio_segment = audio_info.segment

                print("{0}: audio file: {1}, label: {2}, session: {3}, node id: {4}, segment: {5}".format(
                    self.current_index,
                    audio_file,
                    audio_label,
                    audio_session,
                    audio_node_id,
                    audio_segment))

                audio = Utils.load_audio_from_file(os.path.join(self.audio_files_path, audio_file), self.sample_rate)

                self.current_index += 1

                return audio, audio_label, audio_session, audio_node_id, audio_segment

    @staticmethod
    def extract_info_from_filename(sound_file):
        audio_name_split = sound_file.split("/")
        assert len(audio_name_split) > 1, "Wrong audio file path"
        audio_name_split = audio_name_split[-1].split("_")

        assert len(audio_name_split) == 3, "Wrong audio file name"
        audio_node_id = re.findall(r"\d+", audio_name_split[0])[0]
        audio_session = re.findall(r"\d+", audio_name_split[1])[0]
        audio_segment = re.findall(r"\d+", audio_name_split[2])[0]

        return audio_node_id, audio_session, audio_segment

    def get_neighbour(self, label, session, node):
        filtered_items = self.data_frame
        filtered_items = filtered_items[filtered_items.activity_label == label]
        filtered_items = filtered_items[filtered_items.session != session]
        filtered_items = filtered_items[filtered_items.node_id != node]
        print("Selecting neighbour randomly from {} samples".format(len(filtered_items)))

        neighbour = filtered_items.sample().iloc[0]
        return neighbour

    def get_opposite(self, label):
        filtered_items = self.data_frame
        filtered_items = filtered_items[filtered_items.activity_label != label]
        print("Selecting opposite randomly from {} samples".format(len(filtered_items)))

        opposite = filtered_items.sample().iloc[0]
        return opposite
