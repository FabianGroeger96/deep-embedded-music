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

    def get_input_pipeline(self):
        audio_pipeline = TripletsInputPipeline(params=self.params)

        return audio_pipeline

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


if __name__ == '__main__':
    tf.test.main()
