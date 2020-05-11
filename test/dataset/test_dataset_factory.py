import os

import tensorflow as tf

from src.dataset.dataset_factory import DatasetFactory
from src.dataset.dcase_dataset import DCASEDataset
from src.dataset.music_dataset import MusicDataset
from src.utils.params import Params


class TestDatasetFactory(tf.test.TestCase):

    def setUp(self):
        # load the parameters from json file
        json_path = os.path.join("/tf/test_environment/", "config", "params.json")
        self.params = Params(json_path)

    def test_factory_creation_dcase(self):
        dataset = DatasetFactory.create_dataset("DCASEDataset", params=self.params)
        self.assertDTypeEqual(dataset, DCASEDataset)

    def test_factory_creation_music(self):
        dataset = DatasetFactory.create_dataset("MusicDataset", params=self.params)
        self.assertDTypeEqual(dataset, MusicDataset)


if __name__ == '__main__':
    tf.test.main()
