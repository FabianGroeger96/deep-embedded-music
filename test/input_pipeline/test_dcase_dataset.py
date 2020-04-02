import os

import tensorflow as tf

from src.input_pipeline.dcase_dataset import DCASEDataset
from src.utils.params import Params


class TestDCASEDataset(tf.test.TestCase):

    def setUp(self):
        experiment_dir = "/opt/project/test_environment/"

        # load the parameters from json file
        json_path = os.path.join(experiment_dir, "config", "params.json")
        self.params = Params(json_path)

    def get_dataset(self):
        dataset = DCASEDataset(
            dataset_path=self.params.audio_files_path,
            fold=self.params.fold,
            sample_rate=self.params.sample_rate,
            sample_size=self.params.sample_size,
            stereo_channels=self.params.stereo_channels,
            to_mono=self.params.to_mono)

        return dataset

    def test_data_frame_iterator(self):
        dataset = self.get_dataset()

        for audio_entry in dataset:
            self.assertNotEqual(audio_entry.file_name, "")
            self.assertNotEqual(audio_entry.label, "")
            self.assertNotEqual(audio_entry.session, "")
            self.assertNotEqual(audio_entry.node_id, "")
            self.assertNotEqual(audio_entry.segment, "")

    def test_data_frame_neighbour(self):
        dataset = self.get_dataset()

        for index, audio_entry in enumerate(dataset):
            neighbour, _ = dataset.get_neighbour(index)

            self.assertEqual(audio_entry.label, neighbour.label)
            self.assertNotEqual(audio_entry.session, neighbour.session)
            self.assertNotEqual(audio_entry.node_id, neighbour.node_id)

    def test_data_frame_opposite(self):
        dataset = self.get_dataset()

        for index, audio_entry in enumerate(dataset):
            opposite, _ = dataset.get_opposite(index)

            self.assertNotEqual(audio_entry.label, opposite.label)

    def test_data_frame_triplets(self):
        dataset = self.get_dataset()

        for index, audio_entry in enumerate(dataset):
            triplets, labels = dataset.get_triplets(index)

            for triplet, label in zip(triplets, labels):
                self.assertEqual(len(triplet), 3)

                anchor_audio, neighbour_audio, opposite_audio = triplet
                anchor_label, neighbour_label, opposite_label = label

                self.assertNotEqual(len(anchor_audio), 0)
                self.assertNotEqual(len(neighbour_audio), 0)
                self.assertNotEqual(len(opposite_audio), 0)

                self.assertEqual(anchor_label, neighbour_label)
                self.assertNotEqual(anchor_label, opposite_label)

            break


if __name__ == '__main__':
    tf.test.main()
