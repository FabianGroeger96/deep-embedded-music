import logging

import tensorflow as tf


class ConvNet1D(tf.keras.Model):
    def __init__(self, embedding_dim, model_name="ConvNet1D"):
        super(ConvNet1D, self).__init__()

        self.conv_1 = tf.keras.layers.Conv1D(64, 2, input_shape=(None, None, None), padding="same", activation="relu")
        self.conv_2 = tf.keras.layers.Conv1D(64, 2, padding="same", activation="relu")
        self.conv_3 = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")
        self.conv_4 = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")

        self.max_pooling = tf.keras.layers.MaxPool1D(2)

        self.dropout = tf.keras.layers.Dropout(0.3)

        self.dense = tf.keras.layers.Dense(embedding_dim, activation=None)
        self.flatten = tf.keras.layers.Flatten()

        self.l2_normalisation = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))

        self.model_name = model_name
        self.logger = logging.getLogger(self.__class__.__name__)

        tf.print(self.conv_1)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        self.logger.debug("Input features shape: {}".format(inputs.shape))
        if len(inputs.shape) == 3:
            self.logger.debug("Model used to predict single channel input.")
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
        features = self.dropout(features)
        # 3. Conv layer
        features = self.conv_3(features)
        features = self.max_pooling(features)
        # 4. Conv layer
        features = self.conv_4(features)
        features = self.max_pooling(features)
        features = self.dropout(features)
        # Embedding layer
        features = self.flatten(features)
        features = self.dense(features)
        # normalisation
        features = self.l2_normalisation(features)

        return features
