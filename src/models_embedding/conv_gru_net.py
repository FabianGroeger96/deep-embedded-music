import tensorflow as tf

from src.models_embedding.base_model import BaseModel
from src.models_embedding.model_factory import ModelFactory


@ModelFactory.register("ConvGRUNet")
class ConvGRUNet(BaseModel):
    """ A 2-dimensional CNN model with an additional GRU layer before the fully connected one. """

    def __init__(self, params, model_name="ConvGRUNet"):
        """
       Initialises the model.
       Calls the initialise method of the super class.

       :param params: the global hyperparameters for initialising the model.
       :param model_name: the name of the model.
       """

        super(ConvGRUNet, self).__init__(params=params, model_name=model_name, expand_dims=True)

        input_shape = (None, None, None, None)
        self.conv_1 = tf.keras.layers.Conv2D(16, (7, 7), padding="same", input_shape=input_shape, activation="relu")
        self.max_pooling_1 = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 1), padding="same")
        self.bn_1 = tf.keras.layers.BatchNormalization()

        self.conv_2 = tf.keras.layers.Conv2D(32, (5, 5), padding="same", activation="relu")
        self.max_pooling_2 = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 1), padding="same")
        self.bn_2 = tf.keras.layers.BatchNormalization()

        self.conv_3 = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")
        self.max_pooling_3 = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 1), padding="same")
        self.bn_3 = tf.keras.layers.BatchNormalization()

        self.conv_4 = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")
        self.max_pooling_4 = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 1), padding="same")
        self.bn_4 = tf.keras.layers.BatchNormalization()

        self.reshape = tf.keras.layers.Reshape((-1, 32))
        self.gru_1 = tf.keras.layers.GRU(32, return_sequences=True)

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

        # 4. Conv layer
        x = self.conv_4(x)
        x = self.max_pooling_4(x)
        if training:
            x = self.bn_4(x)

        # GRU layer
        x = self.reshape(x)
        x = self.gru_1(x)

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

        # 4. Conv layer
        self.log_cnn_layer(self.conv_4, 4)
        self.log_max_pooling_layer(self.max_pooling_4, 4)
        self.logger.info("---")

        # GRU layer
        self.log_gru_layer(self.gru_1, 5)
        self.logger.info("---")
