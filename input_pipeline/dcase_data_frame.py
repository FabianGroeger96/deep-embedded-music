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

        self.logger = logging.getLogger()

        self.logger.debug(self.data_frame.head())

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self.current_index >= len(self.data_frame):
                raise StopIteration
            else:
                audio_info = self.data_frame.iloc[self.current_index]

                audio_file = audio_info.sound_file
                audio_label = audio_info.activity_label
                audio_session = audio_info.session
                audio_node_id = audio_info.node_id
                audio_segment = audio_info.segment

                self.logger.debug("{0}: audio file: {1}, label: {2}, session: {3}, node id: {4}, segment: {5}".format(
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

    def get_neighbour(self, anchor_id, calc_dist: bool = False):
        anchor = self.data_frame.iloc[anchor_id]

        filtered_items = self.data_frame
        filtered_items = filtered_items[filtered_items.activity_label == anchor.activity_label]
        filtered_items = filtered_items[filtered_items.session != anchor.session]
        filtered_items = filtered_items[filtered_items.node_id != anchor.node_id]

        if len(filtered_items) > 0:
            self.logger.debug("Selecting neighbour randomly from {} samples".format(len(filtered_items)))
        else:
            raise ValueError("No valid neighbour found")

        neighbour = filtered_items.sample().iloc[0]

        if calc_dist:
            dist = self.compare_audio(anchor, neighbour)
            self.logger.debug('Normalized distance between anchor and neighbour: {}'.format(math.ceil(dist)))
        else:
            dist = None

        return neighbour, dist

    def get_opposite(self, anchor_id, calc_dist: bool = False):
        anchor = self.data_frame.iloc[anchor_id]

        filtered_items = self.data_frame
        filtered_items = filtered_items[filtered_items.activity_label != anchor.activity_label]

        if len(filtered_items) > 0:
            self.logger.debug("Selecting opposite randomly from {} samples".format(len(filtered_items)))
        else:
            raise ValueError("No valid opposite found")

        opposite = filtered_items.sample().iloc[0]

        if calc_dist:
            dist = self.compare_audio(anchor, opposite)
            self.logger.debug('Normalized distance between anchor and opposite: {}'.format(math.ceil(dist)))
        else:
            dist = None

        return opposite, dist

    def compare_audio(self, audio_1, audio_2):
        # compute MFCC from audio1
        audio_anchor, _ = librosa.load(os.path.join(self.audio_files_path, audio_1.sound_file), sr=self.sample_rate)
        mfcc_anchor = librosa.feature.mfcc(audio_anchor, self.sample_rate)
        # compute MFCC from audio2
        audio_neigh, _ = librosa.load(os.path.join(self.audio_files_path, audio_2.sound_file), sr=self.sample_rate)
        mfcc_neigh = librosa.feature.mfcc(audio_neigh, self.sample_rate)
        # compute distance between mfccs with dynamic-time-wrapping (dtw)
        dist, cost, acc_cost, path = dtw(mfcc_anchor.T, mfcc_neigh.T, dist=lambda x, y: norm(x - y, ord=1))

        return dist
