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

    def log_model(self):
        # log start sequence
        self.logger.info("--- MODEL Architecture '{}' ---".format(self.model_name))
        # log model specific layers
        self.log_model_specific_layers()
        # log base layers
        self.log_dense_layer(self.dense)
        self.logger.info("Flatten layer")
        self.logger.info("L2 normalisation")
        # log end sequence
        self.logger.info("--- MODEL Architecture ---")

    @abstractmethod
    @tf.function
    def forward_pass(self, inputs):
        pass

    @abstractmethod
    def log_model_specific_layers(self):
        pass

    def log_feature_shape(self, name, feature):
        self.logger.debug("{0} features shape: {1}".format(name, feature.shape))

    def log_cnn_layer(self, cnn_layer, layer_number):
        log_message = "layer {0}: CNN layer, name: {1}, filters: {2}, kernels: {3}, padding: {4}, activation: {5}"
        self.logger.info(log_message.format(layer_number, cnn_layer.name, cnn_layer.filters, cnn_layer.kernel_size,
                                            cnn_layer.padding, cnn_layer.activation.__name__))

    def log_max_pooling_layer(self, max_pooling_layer, layer_number):
        self.logger.info(
            "layer {0}: Max pooling layer, name: {1}, pool_size: {2}".format(layer_number, max_pooling_layer.name,
                                                                      max_pooling_layer.pool_size))

    def log_dropout_layer(self, dropout_layer, layer_number):
        self.logger.info(
            "layer {0}: Dropout layer, name: {1}, rate: {2}".format(layer_number, dropout_layer.name,
                                                             dropout_layer.rate))

    def log_gru_layer(self, gru_layer, layer_number):
        self.logger.info(
            "layer {0}: GRU layer, units: {1}, return_sequences: {2}".format(layer_number, gru_layer.units,
                                                                      gru_layer.return_sequences))

    def log_dense_layer(self, dense_layer):
        self.logger.info(
            "Dense layer, units: {0}, activation: {1}".format(dense_layer.units, dense_layer.activation.__name__))
