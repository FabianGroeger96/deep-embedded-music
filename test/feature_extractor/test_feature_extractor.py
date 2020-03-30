import os

import tensorflow as tf

from src.feature_extractor.log_mel_extractor import LogMelExtractor
from src.feature_extractor.mfcc_extractor import MFCCExtractor
from src.utils.params import Params
from src.utils.utils_audio import AudioUtils


class TestFeatureExtractor(tf.test.TestCase):

    def setUp(self):
        self.experiment_dir = "/opt/project/test_environment/"
        self.audio_file_path = "/opt/project/test_environment/audio/DevNode1_ex1_1.wav"
        self.audio_mono = AudioUtils.load_audio_from_file(self.audio_file_path,
                                                          sample_rate=16000,
                                                          stereo_channels=4,
                                                          to_mono=True)
        self.audio_multi = AudioUtils.load_audio_from_file(self.audio_file_path,
                                                           sample_rate=16000,
                                                           stereo_channels=4,
                                                           to_mono=False)

        # load the parameters from json file
        json_path = os.path.join(self.experiment_dir, "config", "params.json")
        self.params = Params(json_path)

    def test_mfcc_extractor_mono(self):
        feature_extractor = MFCCExtractor(params=self.params)
        # get the output shape of the extractor to check the sizes
        frame_size, n_mfcc_bin = feature_extractor.get_output_shape()
        expected_shape = [frame_size, n_mfcc_bin]

        # extract the feature from the audio
        extracted_features = feature_extractor.extract(self.audio_mono)

        self.assertEqual(expected_shape, extracted_features.shape)

    def test_mfcc_extractor_multi(self):
        feature_extractor = MFCCExtractor(params=self.params)
        # get the output shape of the extractor to check the sizes
        frame_size, n_mfcc_bin = feature_extractor.get_output_shape()
        expected_shape = [frame_size, n_mfcc_bin, 4]

        # extract the feature from the audio
        extracted_features = feature_extractor.extract(self.audio_multi)

        self.assertEqual(expected_shape, extracted_features.shape)

    def test_log_mel_extractor_mono(self):
        feature_extractor = LogMelExtractor(params=self.params)
        # get the output shape of the extractor to check the sizes
        frame_size, n_mfcc_bin = feature_extractor.get_output_shape()
        expected_shape = [frame_size, n_mfcc_bin]

        # extract the feature from the audio
        extracted_features = feature_extractor.extract(self.audio_mono)

        self.assertEqual(expected_shape, extracted_features.shape)

    def test_log_mel_extractor_multi(self):
        feature_extractor = LogMelExtractor(params=self.params)
        # get the output shape of the extractor to check the sizes
        frame_size, n_mfcc_bin = feature_extractor.get_output_shape()
        expected_shape = [frame_size, n_mfcc_bin, 4]

        # extract the feature from the audio
        extracted_features = feature_extractor.extract(self.audio_multi)

        self.assertEqual(expected_shape, extracted_features.shape)


if __name__ == '__main__':
    tf.test.main()
