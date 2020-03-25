import logging

import tensorflow as tf


class ConvNet2D(tf.keras.Model):
    def __init__(self, embedding_dim, model_name="ConvNet2D"):
        super(ConvNet2D, self).__init__()

        self.conv_1 = tf.keras.layers.Conv2D(64, (2, 2), input_shape=(None, None, None, None), padding="same",
                                             activation="relu")
        self.conv_2 = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu")

        self.max_pooling = tf.keras.layers.MaxPool2D((2, 2))

        self.dense = tf.keras.layers.Dense(embedding_dim, activation="relu")
        self.flatten = tf.keras.layers.Flatten()

        self.model_name = model_name
        self.logger = logging.getLogger(self.__class__.__name__)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        self.logger.debug("Input features shape: {}".format(inputs.shape))
        if len(inputs.shape) == 3:
            self.logger.debug("Model used to predict single channel input.")
            inputs = tf.expand_dims(inputs, -1)
            self.logger.debug("Input features shape: {}".format(inputs.shape))
            features = self.forward_pass(inputs)

        elif len(inputs.shape) == 4:
            self.logger.debug("Model used to predict multi channel input.")
            # list of outputs from the different channels
            outputs = []
            for i in range(inputs.shape[-1]):
                # extract one audio channel
                channel = tf.squeeze(inputs[:, i])
                feature_channel = self.forward_pass(channel)
                outputs.append(feature_channel)
            # merge audio channels together
            features = tf.keras.layers.concatenate(outputs)
        else:
            raise ValueError("Input has wrong shape.")

        self.logger.debug("Output features shape: {}".format(features.shape))
        return features

    @tf.function
    def forward_pass(self, inputs):
        # 1. Conv layer
        features = self.conv_1(inputs)
        features = self.max_pooling(features)
        # 2. Conv layer
        features = self.conv_2(features)
        features = self.max_pooling(features)
        # Embedding layer
        features = self.dense(features)
        features = self.flatten(features)

        return features
