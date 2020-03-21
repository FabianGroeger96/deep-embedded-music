import logging

import tensorflow as tf


class DenseEncoder(tf.keras.Model):
    def __init__(self, embedding_dim, model_name="DenseEncoder"):
        super(DenseEncoder, self).__init__()
        self.dense = tf.keras.layers.Dense(embedding_dim, activation="relu")

        self.model_name = model_name
        self.logger = logging.getLogger(self.__class__.__name__)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        self.logger.debug("Input features shape: {}".format(inputs.shape))
        if len(inputs.shape) == 3:
            self.logger.debug("Model used to predict single channel input.")
            embedding = self.dense(inputs)
            self.logger.debug("Output features shape: {}".format(embedding.shape))
            return embedding

        elif len(inputs.shape) == 4:
            self.logger.debug("Model used to predict multi channel input.")
            # list of outputs from the different channels
            outputs = []
            for i in range(inputs.shape[-1]):
                # extract one audio channel
                audio_channel = tf.squeeze(inputs[:, i])
                embedding_channel = self.dense(audio_channel)
                outputs.append(embedding_channel)
            # merge audio channels together
            merged = tf.keras.layers.concatenate(outputs)
            self.logger.debug("Output features shape: {}".format(merged.shape))
            return merged
        else:
            raise ValueError("Input has wrong shape.")
