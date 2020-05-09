from abc import ABC
from enum import Enum

import tensorflow as tf

from src.models_embedding.base_model import BaseModel
from src.models_embedding.model_factory import ModelFactory


class ResNetType(Enum):
    """
    Type of ResNet. Determines the type of layers to use for the ResNet implementation.
    TypeI: uses basic blocks as layers.
    TypeII: uses bottle neck as layers.
    """
    TypeI = 0
    TypeII = 1


class ResNet(BaseModel, ABC):
    """ Abstract implementation for creating different ResNet models_embedding. """

    def __init__(self, layer_params, resnet_type, embedding_dim, l2_amount, model_name="ResNet"):
        """
        Initialises a ResNet model.

        :param layer_params: the number of filters in each layer of the model.
        :param resnet_type: the type of ResNet, determines which layers to use.
            TypeI: uses basic blocks as layers.
            TypeII: uses bottle neck as layers.
        :param embedding_dim: the dimension of the last dense layer (embedding layer).
        :param l2_amount: the weight of the l2 regularisation for each layer.
        :param model_name: the name of the model.
        """
        super(ResNet, self).__init__(embedding_dim=embedding_dim, model_name=model_name, expand_dims=True,
                                     l2_amount=l2_amount)

        self.l2_regularization = tf.keras.regularizers.l2(self.l2_amount)

        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same",
                                            bias_regularizer=self.l2_regularization,
                                            kernel_regularizer=self.l2_regularization)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        if resnet_type == ResNetType.TypeI:
            self.layer1 = self.make_basic_block_layer(filter_num=64,
                                                      blocks=layer_params[0],
                                                      l2_amount=self.l2_amount)
            self.layer2 = self.make_basic_block_layer(filter_num=128,
                                                      blocks=layer_params[1],
                                                      stride=2,
                                                      l2_amount=self.l2_amount)
            self.layer3 = self.make_basic_block_layer(filter_num=256,
                                                      blocks=layer_params[2],
                                                      stride=2,
                                                      l2_amount=self.l2_amount)
            self.layer4 = self.make_basic_block_layer(filter_num=512,
                                                      blocks=layer_params[3],
                                                      stride=2,
                                                      l2_amount=self.l2_amount)
        elif resnet_type == ResNetType.TypeII:
            self.layer1 = self.make_bottleneck_layer(filter_num=64,
                                                     blocks=layer_params[0],
                                                     l2_amount=self.l2_amount)
            self.layer2 = self.make_bottleneck_layer(filter_num=128,
                                                     blocks=layer_params[1],
                                                     stride=2,
                                                     l2_amount=self.l2_amount)
            self.layer3 = self.make_bottleneck_layer(filter_num=256,
                                                     blocks=layer_params[2],
                                                     stride=2,
                                                     l2_amount=self.l2_amount)
            self.layer4 = self.make_bottleneck_layer(filter_num=512,
                                                     blocks=layer_params[3],
                                                     stride=2,
                                                     l2_amount=self.l2_amount)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()

    @tf.function
    def forward_pass(self, inputs, training=None):
        """
        The forward pass through the network.

        :param inputs: the input that will be passed through the model.
        :param training: if the model is training, for disabling dropout, batch norm. etc.
        :return: the output of the forward pass.
        """
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        if training:
            x = tf.keras.layers.Dropout(0.5)(x)

        # embedding
        x = self.flatten(x)
        embedding = self.dense(x)

        # l2 normalisation
        embedding = self.l2_normalisation(embedding)

        return embedding

    def log_model_specific_layers(self):
        """
        Logs the specific layers of the model.
        Is used to log the architecture of the model.

        :return: None.
        """
        # TODO log resnet layers
        pass


@ModelFactory.register("ResNet18")
class ResNet18(ResNet):
    """ Concrete implementation of the standard ResNet18 architecture. """
    def __init__(self, embedding_dim, l2_amount=0.1, model_name="ResNet18"):
        super(ResNet18, self).__init__(layer_params=[2, 2, 2, 2], resnet_type=ResNetType.TypeI,
                                       embedding_dim=embedding_dim, l2_amount=l2_amount, model_name=model_name)


@ModelFactory.register("ResNet34")
class ResNet34(ResNet):
    """ Concrete implementation of the standard ResNet34 architecture. """
    def __init__(self, embedding_dim, l2_amount=0.1, model_name="ResNet34"):
        super(ResNet34, self).__init__(layer_params=[3, 4, 6, 3], resnet_type=ResNetType.TypeI,
                                       embedding_dim=embedding_dim, l2_amount=l2_amount, model_name=model_name)


@ModelFactory.register("ResNet50")
class ResNet50(ResNet):
    """ Concrete implementation of the standard ResNet50 architecture. """
    def __init__(self, embedding_dim, l2_amount=0.1, model_name="ResNet50"):
        super(ResNet50, self).__init__(layer_params=[3, 4, 6, 3], resnet_type=ResNetType.TypeII,
                                       embedding_dim=embedding_dim, l2_amount=l2_amount, model_name=model_name)


@ModelFactory.register("ResNet101")
class ResNet101(ResNet):
    """ Concrete implementation of the standard ResNet101 architecture. """
    def __init__(self, embedding_dim, l2_amount=0.1, model_name="ResNet101"):
        super(ResNet101, self).__init__(layer_params=[3, 4, 23, 3], resnet_type=ResNetType.TypeII,
                                        embedding_dim=embedding_dim, l2_amount=l2_amount, model_name=model_name)


@ModelFactory.register("ResNet152")
class ResNet152(ResNet):
    """ Concrete implementation of the standard ResNet152 architecture. """
    def __init__(self, embedding_dim, l2_amount=0.1, model_name="ResNet152"):
        super(ResNet152, self).__init__(layer_params=[3, 8, 36, 3], resnet_type=ResNetType.TypeII,
                                        embedding_dim=embedding_dim, l2_amount=l2_amount, model_name=model_name)
