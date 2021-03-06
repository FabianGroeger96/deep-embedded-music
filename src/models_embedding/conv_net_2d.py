import tensorflow as tf

from src.models_embedding.base_model import BaseModel
from src.models_embedding.model_factory import ModelFactory


@ModelFactory.register("ConvNet2D")
class ConvNet2D(BaseModel):
    """ A simple 2-dimensional CNN model. """

    def __init__(self, params, model_name="ConvNet2D"):
        """
       Initialises the model.
       Calls the initialise method of the super class.

       :param params: the global hyperparameters for initialising the model.
       :param model_name: the name of the model.
       """
        super(ConvNet2D, self).__init__(params=params, model_name=model_name, expand_dims=True)

        input_shape = (None, None, None, None)

        self.conv_1 = tf.keras.layers.Conv2D(32, (3, 3), input_shape=input_shape, activation="relu", padding="same")
        self.max_pooling_1 = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same")
        self.bn_1 = tf.keras.layers.BatchNormalization()

        self.conv_2 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")
        self.max_pooling_2 = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same")
        self.bn_2 = tf.keras.layers.BatchNormalization()

        self.conv_3 = tf.keras.layers.Conv2D(64, (2, 2), activation="relu", padding="same")
        self.max_pooling_3 = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same")
        self.bn_3 = tf.keras.layers.BatchNormalization()

        self.dropout = tf.keras.layers.Dropout(0.3)

    @tf.function
    def forward_pass(self, inputs, training=None):
        """
        The forward pass through the network.

        :param inputs: the input that will be passed through the model.
        :param training: if the model is training, for disabling dropout, batch norm. etc.
        :return: the output of the forward pass.
        """
        # 1. Conv layer
        x = self.conv_1(inputs)
        x = self.max_pooling_1(x)
        if training:
            x = self.bn_1(x)

        # 2. Conv layer
        x = self.conv_2(x)
        x = self.max_pooling_2(x)
        if training:
            x = self.bn_2(x)

        # 3. Conv layer
        x = self.conv_3(x)
        x = self.max_pooling_3(x)
        if training:
            x = self.bn_3(x)

        # Embedding layer
        x = self.flatten(x)
        x = self.dense(x)
        if training:
            x = self.dropout(x)

        # L2 normalisation
        x = self.l2_normalisation(x)

        return x

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
