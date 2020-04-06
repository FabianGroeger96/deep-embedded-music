import tensorflow as tf

from src.models.base_model import BaseModel
from src.models.model_factory import ModelFactory


@ModelFactory.register("ConvNet2D")
class ConvNet2D(BaseModel):
    """ A simple 2-dimensional CNN model. """

    def __init__(self, embedding_dim, batch_normalisation=False, model_name="ConvNet2D"):
        """
       Initialises the model.
       Calls the initialise method of the super class.

       :param embedding_dim: the dimension for the embedding space.
       :param batch_normalisation: if the batches should be normalised
       :param model_name: the name of the model.
       """
        super(ConvNet2D, self).__init__(embedding_dim=embedding_dim, model_name=model_name, expand_dims=True)

        input_shape = (None, None, None, None)

        self.conv_1 = tf.keras.layers.Conv2D(32, (3, 3), input_shape=input_shape, activation="relu")
        self.max_pooling_1 = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same")

        self.conv_2 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")
        self.max_pooling_2 = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same")

        self.conv_3 = tf.keras.layers.Conv2D(64, (2, 2), activation="relu")
        self.max_pooling_3 = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same")

        self.use_batch_normalisation = batch_normalisation
        if self.use_batch_normalisation:
            self.batch_normalisation = tf.keras.layers.BatchNormalization()

        self.dropout = tf.keras.layers.Dropout(0.3)

    @tf.function
    def forward_pass(self, inputs):
        """
        The forward pass through the network.

        :param inputs: the input that will be passed through the model.
        :return: the output of the forward pass.
        """
        # 1. Conv layer
        features = self.conv_1(inputs)
        self.logger.debug("Feature shape: {}".format(features.shape))
        features = self.max_pooling_1(features)
        self.logger.debug("Feature shape: {}".format(features.shape))
        if self.use_batch_normalisation:
            features = self.batch_normalisation(features)
        # 2. Conv layer
        features = self.conv_2(features)
        self.logger.debug("Feature shape: {}".format(features.shape))
        features = self.max_pooling_2(features)
        self.logger.debug("Feature shape: {}".format(features.shape))
        if self.use_batch_normalisation:
            features = self.batch_normalisation(features)
        # 3. Conv layer
        features = self.conv_3(features)
        self.logger.debug("Feature shape: {}".format(features.shape))
        features = self.max_pooling_3(features)
        self.logger.debug("Feature shape: {}".format(features.shape))
        if self.use_batch_normalisation:
            features = self.batch_normalisation(features)
        # Embedding layer
        features = self.flatten(features)
        self.logger.debug("Feature shape: {}".format(features.shape))
        features = self.dense(features)
        self.logger.debug("Feature shape: {}".format(features.shape))
        features = self.dropout(features)
        self.logger.debug("Feature shape: {}".format(features.shape))
        # L2 normalisation
        features = self.l2_normalisation(features)
        self.logger.debug("Feature shape: {}".format(features.shape))

        return features

    def log_model_specific_layers(self):
        """
        Logs the specific layers of the model.
        Is used to log the architecture of the model.

        :return: None.
        """
        # 1. Conv layer
        self.log_cnn_layer(self.conv_1, 1)
        self.log_max_pooling_layer(self.max_pooling_1, 1)
        self.logger.info("---")

        # 2. Conv layer
        self.log_cnn_layer(self.conv_2, 2)
        self.log_max_pooling_layer(self.max_pooling_2, 2)
        self.logger.info("---")

        # 3. Conv layer
        self.log_cnn_layer(self.conv_3, 3)
        self.log_max_pooling_layer(self.max_pooling_3, 3)
        self.logger.info("---")
