import os

import tensorflow as tf

from src.input_pipeline.dcase_dataset import DCASEDataset
from src.input_pipeline.triplet_input_pipeline import TripletsInputPipeline
from src.utils.params import Params


class TestDCASEDataset(tf.test.TestCase):

    def setUp(self):
        experiment_dir = "/opt/project/test_environment/"

        # load the parameters from json file
        json_path = os.path.join(experiment_dir, "config", "params.json")
        self.params = Params(json_path)

    def test_data_frame_iterator(self):
        audio_iterator = DCASEDataset(
            dataset_path=self.params.audio_files_path,
            fold=self.params.fold,
            sample_rate=self.params.sample_rate)

        for audio_entry in audio_iterator:
            self.assertNotEqual(audio_entry.file_name, "")
            self.assertNotEqual(audio_entry.label, "")
            self.assertNotEqual(audio_entry.session, "")
            self.assertNotEqual(audio_entry.node_id, "")
            self.assertNotEqual(audio_entry.segment, "")

    def test_data_frame_neighbour(self):
        audio_iterator = DCASEDataset(
            dataset_path=self.params.audio_files_path,
            fold=self.params.fold,
            sample_rate=self.params.sample_rate)

        for index, audio_entry in enumerate(audio_iterator):
            neighbour, _ = audio_iterator.get_neighbour(index)

            self.assertEqual(audio_entry.label, neighbour.label)
            self.assertNotEqual(audio_entry.session, neighbour.session)
            self.assertNotEqual(audio_entry.node_id, neighbour.node_id)

    def test_data_frame_opposite(self):
        audio_iterator = DCASEDataset(
            dataset_path=self.params.audio_files_path,
            fold=self.params.fold,
            sample_rate=self.params.sample_rate)

        for index, audio_entry in enumerate(audio_iterator):
            opposite, _ = audio_iterator.get_opposite(index)

            self.assertNotEqual(audio_entry.label, opposite.label)


if __name__ == '__main__':
    tf.test.main()
