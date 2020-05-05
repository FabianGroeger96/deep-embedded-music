import logging

import tensorflow as tf


class ClassifierLogistic(tf.keras.Model):
    """ The logistic classifier. It will be used to evaluate the embedding space. """

    def __init__(self, model_name, n_labels):
        """
        Initialises the model.

        :param model_name: the name of the model.
        :param n_labels: the count of classes to predict.
        """
        super(ClassifierLogistic, self).__init__()

        self.logger = logging.getLogger(model_name)
        self.model_name = model_name

        self.dense_output = tf.keras.layers.Dense(n_labels, activation="softmax", name="output")

    @tf.function
    def call(self, inputs):
        """
        Executes a forward pass through the model.
        Based on the input, it will execute the forward pass for a single channel or for multiple channels.
        Will be executed as a graph (@tf.function).

        :param inputs: the input that will be passed through the model.
        :return: returns the output of the model.
        :raises: ValueError: if the input has the wrong shape.
        """
        if len(inputs.shape) == 3:
            inputs = tf.keras.layers.Flatten()(inputs)

        features = self.dense_output(inputs)

        return features
