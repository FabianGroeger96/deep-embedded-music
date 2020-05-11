import os

import tensorflow as tf

from src.feature_extractor.extractor_factory import ExtractorFactory
from src.feature_extractor.log_mel_extractor import LogMelExtractor
from src.feature_extractor.mfcc_extractor import MFCCExtractor
from src.utils.params import Params


class TestFeatureExtractorFactory(tf.test.TestCase):

    def setUp(self):
        # load the parameters from json file
        json_path = os.path.join("/tf/test_environment/", "config", "params.json")
        self.params = Params(json_path)

    def test_factory_creation_logmel(self):
        dataset = ExtractorFactory.create_extractor("LogMelExtractor", params=self.params)
        self.assertDTypeEqual(dataset, LogMelExtractor)

    def test_factory_creation_mfcc(self):
        dataset = ExtractorFactory.create_extractor("MFCCExtractor", params=self.params)
        self.assertDTypeEqual(dataset, MFCCExtractor)


if __name__ == '__main__':
    tf.test.main()
