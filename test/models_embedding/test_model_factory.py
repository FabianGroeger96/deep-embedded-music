import os

import tensorflow as tf

from src.models_embedding.conv_net_2d import ConvNet2D
from src.models_embedding.model_factory import ModelFactory
from src.models_embedding.resnet import ResNet18
from src.utils.params import Params


class TestModelFactory(tf.test.TestCase):

    def setUp(self):
        # load the parameters from json file
        json_path = os.path.join("/tf/test_environment/", "config", "params.json")
        self.params = Params(json_path)

    def test_factory_creation_resnet(self):
        dataset = ModelFactory.create_model("ResNet18", params=self.params)
        self.assertDTypeEqual(dataset, ResNet18)

    def test_factory_creation_convnet2d(self):
        dataset = ModelFactory.create_model("ConvNet2D", params=self.params)
        self.assertDTypeEqual(dataset, ConvNet2D)


if __name__ == '__main__':
    tf.test.main()
