import argparse
import os

import tensorflow as tf

from src.feature_extractor.log_mel_extractor import LogMelExtractor
from src.feature_extractor.mfcc_extractor import MFCCExtractor
from src.input_pipeline.dcase_dataset import DCASEDataset
from src.input_pipeline.triplet_input_pipeline import TripletsInputPipeline
from src.utils.params import Params


class TestInputPipeline(tf.test.TestCase):

    def setUp(self):
        experiment_dir = "/opt/project/test/test_environment/"

        # load the parameters from json file
        json_path = os.path.join(experiment_dir, "config", "params.json")
        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
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

    def test_dataset_generator_channels(self):
        # change pipeline to single channel
        self.params.stereo_channels = 1
        self.params.to_mono = True
        audio_pipeline = self.get_input_pipeline()
        dataset_iterator = audio_pipeline.get_dataset(feature_extractor=None, shuffle=True, calc_dist=False)
        for anchor, neighbour, opposite, triplet_labels in dataset_iterator:
            # check if triplets have a third dimension (channel)
            expected_shape = [self.params.batch_size, self.params.sample_rate]
            self.assertEqual(expected_shape, anchor.shape)
            self.assertEqual(expected_shape, neighbour.shape)
            self.assertEqual(expected_shape, opposite.shape)

        # change pipeline to multiple channels
        self.params.stereo_channels = 4
        self.params.to_mono = False
        audio_pipeline = self.get_input_pipeline()
        dataset_iterator = audio_pipeline.get_dataset(feature_extractor=None, shuffle=True, calc_dist=False)
        for anchor, neighbour, opposite, triplet_labels in dataset_iterator:
            # check if triplets have a third dimension (channel)
            expected_shape = [self.params.batch_size, self.params.sample_rate, self.params.stereo_channels]
            self.assertEqual(expected_shape, anchor.shape)
            self.assertEqual(expected_shape, neighbour.shape)
            self.assertEqual(expected_shape, opposite.shape)

    def test_dataset_generator_batch_size(self):
        audio_pipeline = self.get_input_pipeline()
        dataset_iterator = audio_pipeline.get_dataset(feature_extractor=None, shuffle=True, calc_dist=False)
        for anchor, neighbour, opposite, triplet_labels in dataset_iterator:
            # check if there are the same number of samples as requested (batch_size)
            self.assertEqual(self.params.batch_size, anchor.shape[0])
            self.assertEqual(self.params.batch_size, neighbour.shape[0])
            self.assertEqual(self.params.batch_size, opposite.shape[0])
            self.assertEqual(self.params.batch_size, triplet_labels.shape[0])

    def test_dataset_generator_sample_rate_audios(self):
        audio_pipeline = self.get_input_pipeline()
        dataset_iterator = audio_pipeline.get_dataset(feature_extractor=None, shuffle=True, calc_dist=False)
        for anchor, neighbour, opposite, triplet_labels in dataset_iterator:
            # check if triplets have the correct audio size (sample_rate)
            self.assertEqual(self.params.sample_rate, anchor.shape[1])
            self.assertEqual(self.params.sample_rate, neighbour.shape[1])
            self.assertEqual(self.params.sample_rate, opposite.shape[1])

    def test_dataset_generator_triplets_valid(self):
        audio_pipeline = self.get_input_pipeline()
        dataset_iterator = audio_pipeline.get_dataset(feature_extractor=None, shuffle=True, calc_dist=False)
        for anchor, neighbour, opposite, triplet_labels in dataset_iterator:
            # check if triplets are valid
            for a_label, n_label, o_label in triplet_labels:
                # check if anchor and neighbour have the same labels
                self.assertTrue(a_label == n_label)
                # check if anchor and opposite have different labels
                self.assertTrue(a_label != o_label)

    def test_dataset_generator_mfcc_extractor(self):
        feature_extractor = MFCCExtractor(params=self.params)
        # get the output shape of the extractor to check the sizes
        frame_size, n_mfcc_bin = feature_extractor.get_output_shape()

        audio_pipeline = self.get_input_pipeline()
        datset = audio_pipeline.get_dataset(feature_extractor=feature_extractor, shuffle=True, calc_dist=False)
        for anchor, neighbour, opposite, triplet_labels in datset:
            # check if triplets have the correct shape
            expected_shape = [self.params.batch_size, frame_size, n_mfcc_bin, self.params.stereo_channels]
            self.assertEqual(expected_shape, anchor.shape)
            self.assertEqual(expected_shape, neighbour.shape)
            self.assertEqual(expected_shape, opposite.shape)

    def test_dataset_generator_log_mel_extractor(self):
        feature_extractor = LogMelExtractor(params=self.params)
        # get the output shape of the extractor to check the sizes
        frame_size, n_mel_bin = feature_extractor.get_output_shape()

        audio_pipeline = self.get_input_pipeline()
        dataset = audio_pipeline.get_dataset(feature_extractor=feature_extractor, shuffle=True, calc_dist=False)
        for anchor, neighbour, opposite, triplet_labels in dataset:
            # check if triplets have the correct shape
            expected_shape = [self.params.batch_size, frame_size, n_mel_bin, self.params.stereo_channels]
            self.assertEqual(expected_shape, anchor.shape)
            self.assertEqual(expected_shape, neighbour.shape)
            self.assertEqual(expected_shape, opposite.shape)


if __name__ == '__main__':
    tf.test.main()
