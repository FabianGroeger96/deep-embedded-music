import logging
from abc import ABC, abstractmethod

import tensorflow as tf


class BaseModel(tf.keras.Model, ABC):
    """ The abstract base model. All models will inherit this base. """

    def __init__(self, embedding_dim, model_name, expand_dims):
        """
        Initialises the model.
        :param embedding_dim: the dimension for the embedding space.
        :param model_name: the name of the model.
        :param expand_dims: if the model should expand its dimension.
        """
        super(BaseModel, self).__init__()

        self.logger = logging.getLogger(model_name)

        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.expand_dims = expand_dims

        self.dense = tf.keras.layers.Dense(embedding_dim, activation=None)
        self.flatten = tf.keras.layers.Flatten()

        self.l2_normalisation = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))

    @tf.function
    def call(self, inputs, training=None):
        """
        Executes a forward pass through the model.
        Based on the input, it will execute the forward pass for a single channel or for multiple channels.
        Will be executed as a graph (@tf.function).
        :param inputs: the input that will be passed through the model.
        :param training: if the model is training.
        :return: returns the output of the model.
        :raises: ValueError: if the input has the wrong shape.
        """
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
        """
        Logs the architecture of the model.
        :return: None.
        """
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
        """
        Abstract method.
        The forward pass through the network.
        Needs to be implemented in models which inherit the BaseModel.
        :param inputs: the input that will be passed through the model.
        :return: the output of the forward pass.
        """
        pass

    @abstractmethod
    def log_model_specific_layers(self):
        """
        Abstract method.
        Logs the specific layers of the inherited model.
        Is used to log the architecture of the model.
        Needs to be implemented in models which inherit the BaseModel.
        :return: None.
        """
        pass

    def log_feature_shape(self, name, feature):
        """
        Logs the shape a feature.
        :param name: the name of the feature.
        :param feature: the feature to log its shape.
        :return: None.
        """
        self.logger.debug("{0} features shape: {1}".format(name, feature.shape))

    def log_cnn_layer(self, cnn_layer, layer_number):
        """
        Logs the important attributes of a CNN layer.
        :param cnn_layer: the cnn layer to log.
        :param layer_number: the layer number within the entire model.
        :return: None.
        """
        log_message = "layer {0}: CNN layer, name: {1}, filters: {2}, kernels: {3}, padding: {4}, activation: {5}"
        self.logger.info(log_message.format(layer_number, cnn_layer.name, cnn_layer.filters, cnn_layer.kernel_size,
                                            cnn_layer.padding, cnn_layer.activation.__name__))

    def log_max_pooling_layer(self, max_pooling_layer, layer_number):
        """
        Logs the important attributes of a Max pooling layer.
        :param max_pooling_layer: the max pooling layer to log.
        :param layer_number: the layer number within the entire model.
        :return: None.
        """
        self.logger.info(
            "layer {0}: Max pooling layer, name: {1}, pool_size: {2}".format(layer_number, max_pooling_layer.name,
                                                                             max_pooling_layer.pool_size))

    def log_dropout_layer(self, dropout_layer, layer_number):
        """
        Logs the important attributes of a Dropout layer.
        :param dropout_layer: the dropout layer to log.
        :param layer_number: the layer number within the entire model.
        :return: None.
        """
        self.logger.info(
            "layer {0}: Dropout layer, name: {1}, rate: {2}".format(layer_number, dropout_layer.name,
                                                                    dropout_layer.rate))

    def log_gru_layer(self, gru_layer, layer_number):
        """
        Logs the important attributes of a GRU layer.
        :param gru_layer: the GRU layer to log.
        :param layer_number: the layer number within the entire model.
        :return: None.
        """
        self.logger.info(
            "layer {0}: GRU layer, units: {1}, return_sequences: {2}".format(layer_number, gru_layer.units,
                                                                             gru_layer.return_sequences))

    def log_dense_layer(self, dense_layer):
        """
        Logs the important attributes of a Dense layer.
        :param dense_layer: the dense layer to log.
        :return: None.
        """
        self.logger.info(
            "Dense layer, units: {0}, activation: {1}".format(dense_layer.units, dense_layer.activation.__name__))
