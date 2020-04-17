import tensorflow as tf

from src.models.base_model import BaseModel
from src.models.model_factory import ModelFactory


@ModelFactory.register("ConvNet1D")
class ConvNet1D(BaseModel):
    """ A simple 1-dimensional CNN model. """

    def __init__(self, embedding_dim, model_name="ConvNet1D"):
        """
        Initialises the model.
        Calls the initialise method of the super class.

        :param embedding_dim: the dimension for the embedding space.
        :param model_name: the name of the model.
        """
        super(ConvNet1D, self).__init__(embedding_dim=embedding_dim, model_name=model_name, expand_dims=False)

        input_shape = (None, None, None)
        self.conv_1 = tf.keras.layers.Conv1D(64, 2, input_shape=input_shape, padding="same", activation="relu")
        self.conv_2 = tf.keras.layers.Conv1D(64, 2, padding="same", activation="relu")
        self.conv_3 = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")
        self.conv_4 = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")

        self.max_pooling = tf.keras.layers.MaxPool1D(2)

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
        x = self.max_pooling(x)
        if training:
            x = self.dropout(x)

        # 2. Conv layer
        x = self.conv_2(x)
        x = self.max_pooling(x)
        if training:
            x = self.dropout(x)

        # 3. Conv layer
        x = self.conv_3(x)
        x = self.max_pooling(x)
        if training:
            x = self.dropout(x)

        # 4. Conv layer
        x = self.conv_4(x)
        x = self.max_pooling(x)
        if training:
            x = self.dropout(x)

        # Embedding layer
        x = self.flatten(x)
        x = self.dense(x)

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
        self.log_max_pooling_layer(self.max_pooling, 1)
        self.log_dropout_layer(self.dropout, 1)
        self.logger.info("---")

        # 2. Conv layer
        self.log_cnn_layer(self.conv_2, 2)
        self.log_max_pooling_layer(self.max_pooling, 2)
        self.log_dropout_layer(self.dropout, 2)
        self.logger.info("---")

        # 3. Conv layer
        self.log_cnn_layer(self.conv_3, 3)
        self.log_max_pooling_layer(self.max_pooling, 3)
        self.log_dropout_layer(self.dropout, 3)
        self.logger.info("---")

        # 4. Conv layer
        self.log_cnn_layer(self.conv_4, 4)
        self.log_max_pooling_layer(self.max_pooling, 4)
        self.log_dropout_layer(self.dropout, 4)
        self.logger.info("---")
