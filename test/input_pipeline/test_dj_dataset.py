import os

import tensorflow as tf

from src.input_pipeline.dj_dataset import DJDataset
from src.utils.params import Params


class TestDJDataset(tf.test.TestCase):

    def setUp(self):
        self.dataset_dir = "/opt/project/data/dj-set/MusicDataset/"

        # load the parameters from json file
        json_path = os.path.join("/opt/project/test_environment/", "config", "params.json")
        self.params = Params(json_path)

    def test_data_frame_iterator(self):
        audio_iterator = DJDataset(
            dataset_path=self.dataset_dir,
            sample_rate=self.params.sample_rate)

        for audio_entry in audio_iterator:
            self.assertNotEqual(audio_entry.name, "")
            self.assertNotEqual(audio_entry.path, "")
            self.assertNotEqual(audio_entry.label, "")

    def test_data_frame_neighbour(self):
        audio_iterator = DJDataset(
            dataset_path=self.dataset_dir,
            sample_rate=self.params.sample_rate)

        for index, audio_entry in enumerate(audio_iterator):
            neighbour, _ = audio_iterator.get_neighbour(index)

            self.assertNotEqual(audio_entry.name, neighbour.name)
            self.assertEqual(audio_entry.label, neighbour.label)

    def test_data_frame_opposite(self):
        audio_iterator = DJDataset(
            dataset_path=self.dataset_dir,
            sample_rate=self.params.sample_rate)

        for index, audio_entry in enumerate(audio_iterator):
            opposite, _ = audio_iterator.get_opposite(index)

            self.assertNotEqual(audio_entry.label, opposite.label)

    def test_data_frame_triplets(self):
        audio_iterator = DJDataset(
            dataset_path=self.dataset_dir,
            sample_rate=self.params.sample_rate)

        for index, audio_entry in enumerate(audio_iterator):
            neighbour, _ = audio_iterator.get_neighbour(index)
            opposite, _ = audio_iterator.get_opposite(index)

            triplets = audio_iterator.get_triplets(audio_entry, neighbour, opposite)

            for triplet in triplets:
                self.assertNotEqual(len(triplet[0]), 0)
                self.assertNotEqual(len(triplet[1]), 0)
                self.assertNotEqual(len(triplet[2]), 0)

            break


if __name__ == '__main__':
    tf.test.main()
