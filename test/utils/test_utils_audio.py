import os

import tensorflow as tf

from src.feature_extractor.log_mel_extractor import LogMelExtractor
from src.feature_extractor.mfcc_extractor import MFCCExtractor
from src.input_pipeline.triplet_input_pipeline import TripletsInputPipeline
from src.utils.params import Params
from src.utils.utils_audio import AudioUtils


class TestUtilsAudio(tf.test.TestCase):

    def setUp(self):
        self.audio_file_path = "/opt/project/test_environment/audio/DevNode1_ex1_1.wav"

    def test_audio_loading_mono(self):
        expected_shape = (16000,)
        audio = AudioUtils.load_audio_from_file(self.audio_file_path,
                                                sample_rate=16000,
                                                stereo_channels=4,
                                                to_mono=True)

        self.assertEqual(expected_shape, audio.shape)

    def test_audio_loading_multi_channel(self):
        expected_shape = (16000, 4)
        audio = AudioUtils.load_audio_from_file(self.audio_file_path,
                                                sample_rate=16000,
                                                stereo_channels=4,
                                                to_mono=False)

        self.assertEqual(expected_shape, audio.shape)


if __name__ == '__main__':
    tf.test.main()
