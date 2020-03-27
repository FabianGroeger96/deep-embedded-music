import logging
from abc import ABC, abstractmethod

import tensorflow as tf


class BaseModel(tf.keras.Model, ABC):
    def __init__(self, embedding_dim, model_name, expand_dims):
        super(BaseModel, self).__init__()

        self.logger = logging.getLogger(model_name)

        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.expand_dims = expand_dims

        self.dense = tf.keras.layers.Dense(embedding_dim, activation=None)
        self.flatten = tf.keras.layers.Flatten()

        self.l2_normalisation = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))

    @tf.function
    def call(self, inputs, training=None, mask=None):
        self.log_feature_shape("Inputs", inputs)
        if len(inputs.shape) == 3:
            self.logger.debug("Model used to predict single channel input.")
            # expands dimension to work with 2D inputs
            if self.expand_dims:
                inputs = tf.expand_dims(inputs, -1)
            features = self.forward_pass(inputs)

        elif len(inputs.shape) == 4:
            self.logger.debug("Model used to predict multi channel input.")
            # expands dimension to work with 2D inputs
            if self.expand_dims:
                inputs = tf.expand_dims(inputs, -1)
            # list of outputs from the different channels
            outputs = []
            for i in range(inputs.shape[-1]):
                # extract one audio channel
                audio_channel = tf.squeeze(inputs[:, i])
                feature_channel = self.forward_pass(audio_channel)
                outputs.append(feature_channel)
            # merge audio channels together
            features = tf.keras.layers.concatenate(outputs)
        else:
            raise ValueError("Input has wrong shape.")

        self.log_feature_shape("Output", features)
        return features

    @abstractmethod
    @tf.function
    def forward_pass(self, inputs):
        pass

    def log_feature_shape(self, name, feature):
        self.logger.debug("{0} features shape: {1}".format(name, feature.shape))
