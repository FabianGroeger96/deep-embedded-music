import os

import librosa
import tensorflow as tf

from src.input_pipeline.music_dataset import MusicDataset
from src.utils.params import Params


class TestMusicDataset(tf.test.TestCase):

    def setUp(self):
        # load the parameters from json file
        json_path = os.path.join("/opt/project/test_environment/", "config", "params.json")
        self.params = Params(json_path)

    def get_dataset(self):
        dataset = MusicDataset(params=self.params)

        return dataset

    def test_data_frame_iterator(self):
        dataset = self.get_dataset()

        for audio_entry in dataset:
            self.assertNotEqual(audio_entry.file_name, "")

    def test_data_frame_triplets(self):
        dataset = self.get_dataset()

        for index, audio_entry in enumerate(dataset):
            anchor = dataset.df_train.iloc[index]
            anchor_audio, _ = librosa.load(anchor.file_name, self.params.sample_rate)
            anchor_audio_length = int(len(anchor_audio) / self.params.sample_rate)

            dataset.fill_opposite_selection(index)

            triplets = dataset.get_triplets(index, audio_length=anchor_audio_length)
            for triplet in triplets:
                self.assertEqual(len(triplet), 3)

                anchor_seg, neighbour_seg, opposite_seg = triplet

                self.assertEqual(len(anchor_seg), 2)
                self.assertEqual(len(neighbour_seg), 2)
                self.assertEqual(len(opposite_seg), 2)

            break


if __name__ == '__main__':
    tf.test.main()
