import tensorflow as tf

from src.models.base_model import BaseModel
from src.models.model_factory import ModelFactory


@ModelFactory.register("ConvNet2D")
class ConvNet2D(BaseModel):
    def __init__(self, embedding_dim, model_name="ConvNet2D"):
        super(ConvNet2D, self).__init__(embedding_dim=embedding_dim, model_name=model_name, expand_dims=True)

        input_shape = (None, None, None, None)
        self.conv_1 = tf.keras.layers.Conv2D(64, 2, input_shape=input_shape, padding="same", activation="relu")
        self.conv_2 = tf.keras.layers.Conv2D(64, 2, padding="same", activation="relu")
        self.conv_3 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")
        self.conv_4 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")

        self.max_pooling = tf.keras.layers.MaxPool2D(2)

        self.dropout = tf.keras.layers.Dropout(0.3)

    @tf.function
    def forward_pass(self, inputs):
        # 1. Conv layer
        features = self.conv_1(inputs)
        features = self.dropout(features)
        # 2. Conv layer
        features = self.conv_2(features)
        features = self.max_pooling(features)
        features = self.dropout(features)
        # 3. Conv layer
        features = self.conv_3(features)
        features = self.dropout(features)
        # 4. Conv layer
        features = self.conv_4(features)
        features = self.max_pooling(features)
        features = self.dropout(features)
        # Embedding layer
        features = self.flatten(features)
        features = self.dense(features)
        # L2 normalisation
        features = self.l2_normalisation(features)

        return features

    def log_model_specific_layers(self):
        # 1. Conv layer
        self.log_cnn_layer(self.conv_1, 1)
        self.log_dropout_layer(self.dropout, 1)
        self.logger.info("---")

        # 2. Conv layer
        self.log_cnn_layer(self.conv_2, 2)
        self.log_max_pooling_layer(self.max_pooling, 2)
        self.log_dropout_layer(self.dropout, 2)
        self.logger.info("---")

        # 3. Conv layer
        self.log_cnn_layer(self.conv_3, 3)
        self.log_dropout_layer(self.dropout, 3)
        self.logger.info("---")

        # 4. Conv layer
        self.log_cnn_layer(self.conv_4, 4)
        self.log_max_pooling_layer(self.max_pooling, 4)
        self.log_dropout_layer(self.dropout, 4)
        self.logger.info("---")
