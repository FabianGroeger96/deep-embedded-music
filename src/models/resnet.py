from abc import ABC
from enum import Enum

import tensorflow as tf

from src.models.base_model import BaseModel
from src.models.model_factory import ModelFactory


class ResNetType(Enum):
    TypeI = 0
    TypeII = 1


class ResNet(BaseModel, ABC):
    def __init__(self, layer_params, resnet_type, embedding_dim, l2_amount, model_name="ResNet"):
        super(ResNet, self).__init__(embedding_dim=embedding_dim, model_name=model_name, expand_dims=True,
                                     l2_amount=l2_amount)

        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same",
                                            bias_regularizer=tf.keras.regularizers.l2(self.l2_amount),
                                            kernel_regularizer=tf.keras.regularizers.l2(self.l2_amount))
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
    def __init__(self, embedding_dim, l2_amount=0.1, model_name="ResNet18"):
        super(ResNet18, self).__init__(layer_params=[2, 2, 2, 2], resnet_type=ResNetType.TypeI,
                                       embedding_dim=embedding_dim, l2_amount=l2_amount, model_name=model_name)


@ModelFactory.register("ResNet34")
class ResNet34(ResNet):
    def __init__(self, embedding_dim, l2_amount=0.1, model_name="ResNet34"):
        super(ResNet34, self).__init__(layer_params=[3, 4, 6, 3], resnet_type=ResNetType.TypeI,
                                       embedding_dim=embedding_dim, l2_amount=l2_amount, model_name=model_name)


@ModelFactory.register("ResNet50")
class ResNet50(ResNet):
    def __init__(self, embedding_dim, l2_amount=0.1, model_name="ResNet50"):
        super(ResNet50, self).__init__(layer_params=[3, 4, 6, 3], resnet_type=ResNetType.TypeII,
                                       embedding_dim=embedding_dim, l2_amount=l2_amount, model_name=model_name)


@ModelFactory.register("ResNet101")
class ResNet101(ResNet):
    def __init__(self, embedding_dim, l2_amount=0.1, model_name="ResNet101"):
        super(ResNet101, self).__init__(layer_params=[3, 4, 23, 3], resnet_type=ResNetType.TypeII,
                                        embedding_dim=embedding_dim, l2_amount=l2_amount, model_name=model_name)


@ModelFactory.register("ResNet152")
class ResNet152(ResNet):
    def __init__(self, embedding_dim, l2_amount=0.1, model_name="ResNet152"):
        super(ResNet152, self).__init__(layer_params=[3, 8, 36, 3], resnet_type=ResNetType.TypeII,
                                        embedding_dim=embedding_dim, l2_amount=l2_amount, model_name=model_name)
