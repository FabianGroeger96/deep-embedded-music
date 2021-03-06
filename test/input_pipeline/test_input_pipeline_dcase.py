import os

import tensorflow as tf
import numpy as np

from src.feature_extractor.log_mel_extractor import LogMelExtractor
from src.feature_extractor.mfcc_extractor import MFCCExtractor
from src.dataset.dataset_factory import DatasetFactory
from src.input_pipeline.triplet_input_pipeline import TripletsInputPipeline
from src.utils.params import Params


class TestInputPipeline(tf.test.TestCase):

    def setUp(self):
        # load the parameters from json file
        json_path = os.path.join("/tf/test_environment/", "config", "params.json")
        self.params = Params(json_path)

        self.set_dcase_dataset()

    def get_input_pipeline(self):
        audio_pipeline = TripletsInputPipeline(params=self.params, dataset=self.dataset, log=True)

        return audio_pipeline

    def set_dcase_dataset(self):
        self.params.dataset = "DCASEDataset"
        self.dataset = DatasetFactory.create_dataset("DCASEDataset", params=self.params)

    def test_dataset_generator_channels(self):
        # change pipeline to single channel
        self.params.stereo_channels = 1
        self.params.to_mono = True
        self.set_dcase_dataset()
        # instantiate input pipeline
        audio_pipeline = self.get_input_pipeline()
        dataset_iterator = audio_pipeline.get_dataset(feature_extractor=None, shuffle=True)
        for anchor, neighbour, opposite, _ in dataset_iterator:
            # check if triplets have a third dimension (channel)
            expected_shape = [self.params.batch_size, self.params.sample_tile_size * self.params.sample_rate]
            self.assertEqual(expected_shape, anchor.shape)
            self.assertEqual(expected_shape, neighbour.shape)
            self.assertEqual(expected_shape, opposite.shape)

            break

        # change pipeline to multiple channels
        self.params.stereo_channels = 4
        self.params.to_mono = False
        self.set_dcase_dataset()
        # instantiate input pipeline
        audio_pipeline = self.get_input_pipeline()
        dataset_iterator = audio_pipeline.get_dataset(feature_extractor=None, shuffle=True)
        for anchor, neighbour, opposite, _ in dataset_iterator:
            # check if triplets have a third dimension (channel)
            expected_shape = [self.params.batch_size, self.params.sample_tile_size * self.params.sample_rate,
                              self.params.stereo_channels]
            self.assertEqual(expected_shape, anchor.shape)
            self.assertEqual(expected_shape, neighbour.shape)
            self.assertEqual(expected_shape, opposite.shape)

            break

    def test_dataset_generator_batch_size(self):
        self.set_dcase_dataset()
        audio_pipeline = self.get_input_pipeline()
        dataset_iterator = audio_pipeline.get_dataset(feature_extractor=None, shuffle=True)
        for anchor, neighbour, opposite, _ in dataset_iterator:
            # check if there are the same number of samples as requested (batch_size)
            self.assertEqual(self.params.batch_size, anchor.shape[0])
            self.assertEqual(self.params.batch_size, neighbour.shape[0])
            self.assertEqual(self.params.batch_size, opposite.shape[0])

            break

    def test_dataset_generator_sample_rate_audios(self):
        self.set_dcase_dataset()
        audio_pipeline = self.get_input_pipeline()
        dataset_iterator = audio_pipeline.get_dataset(feature_extractor=None, shuffle=True)
        for anchor, neighbour, opposite, _ in dataset_iterator:
            # check if triplets have the correct audio size (sample_rate)
            self.assertEqual(self.params.sample_rate * self.params.sample_tile_size, anchor.shape[1])
            self.assertEqual(self.params.sample_rate * self.params.sample_tile_size, neighbour.shape[1])
            self.assertEqual(self.params.sample_rate * self.params.sample_tile_size, opposite.shape[1])

            self.assertFalse(np.array_equal(anchor.numpy(), neighbour.numpy()))
            self.assertFalse(np.array_equal(anchor.numpy(), opposite.numpy()))
            self.assertFalse(np.array_equal(neighbour.numpy(), opposite.numpy()))

            break

    def test_dataset_generator_not_equal_segments(self):
        self.set_dcase_dataset()
        audio_pipeline = self.get_input_pipeline()
        dataset_iterator = audio_pipeline.get_dataset(feature_extractor=None, shuffle=True)
        for anchor, neighbour, opposite, _ in dataset_iterator:
            self.assertFalse(np.array_equal(anchor.numpy(), neighbour.numpy()))
            self.assertFalse(np.array_equal(anchor.numpy(), opposite.numpy()))
            self.assertFalse(np.array_equal(neighbour.numpy(), opposite.numpy()))

            break

    def test_dataset_generator_mfcc_extractor(self):
        feature_extractor = MFCCExtractor(params=self.params)
        # get the output shape of the extractor to check the sizes
        frame_size, n_mfcc_bin = feature_extractor.get_output_shape()

        # change pipeline to multiple channels
        self.params.stereo_channels = 4
        self.params.to_mono = False
        self.set_dcase_dataset()

        audio_pipeline = self.get_input_pipeline()
        datset = audio_pipeline.get_dataset(feature_extractor=feature_extractor, shuffle=True)
        for anchor, neighbour, opposite, _ in datset:
            # check if triplets have the correct shape
            expected_shape = [self.params.batch_size, frame_size, n_mfcc_bin, self.params.stereo_channels]
            self.assertEqual(expected_shape, anchor.shape)
            self.assertEqual(expected_shape, neighbour.shape)
            self.assertEqual(expected_shape, opposite.shape)

    def test_dataset_generator_log_mel_extractor(self):
        feature_extractor = LogMelExtractor(params=self.params)
        # get the output shape of the extractor to check the sizes
        frame_size, n_mel_bin = feature_extractor.get_output_shape()

        # change pipeline to multiple channels
        self.params.stereo_channels = 4
        self.params.to_mono = False
        self.set_dcase_dataset()

        audio_pipeline = self.get_input_pipeline()
        dataset = audio_pipeline.get_dataset(feature_extractor=feature_extractor, shuffle=True)
        for anchor, neighbour, opposite, _ in dataset:
            # check if triplets have the correct shape
            expected_shape = [self.params.batch_size, frame_size, n_mel_bin, self.params.stereo_channels]
            self.assertEqual(expected_shape, anchor.shape)
            self.assertEqual(expected_shape, neighbour.shape)
            self.assertEqual(expected_shape, opposite.shape)


if __name__ == '__main__':
    tf.test.main()
