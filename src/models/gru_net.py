import tensorflow as tf

from src.models.base_model import BaseModel
from src.models.model_factory import ModelFactory


@ModelFactory.register("GRUNet")
class GRUNet(BaseModel):
    """ A simple 1-dimensional GRU model. """

    def __init__(self, embedding_dim, model_name="GRUNet"):
        """
       Initialises the model.
       Calls the initialise method of the super class.

       :param embedding_dim: the dimension for the embedding space.
       :param model_name: the name of the model.
       """
        super(GRUNet, self).__init__(embedding_dim=embedding_dim, model_name=model_name, expand_dims=False)

        self.gru_1 = tf.keras.layers.GRU(64, return_sequences=True)
        self.gru_2 = tf.keras.layers.GRU(128)

    @tf.function
    def forward_pass(self, inputs):
        """
        The forward pass through the network.

        :param inputs: the input that will be passed through the model.
        :return: the output of the forward pass.
        """
        # 1. GRU layer
        features = self.gru_1(inputs)
        # 2. GRU layer
        features = self.gru_2(features)
        # Embedding layer
        features = self.flatten(features)
        features = self.dense(features)
        # L2 normalisation
        features = self.l2_normalisation(features)

        return features

    def log_model_specific_layers(self):
        """
        Logs the specific layers of the model.
        Is used to log the architecture of the model.

        :return: None.
        """
        # 1. GRU layer
        self.log_gru_layer(self.gru_1, layer_number=1)
        self.logger.info("---")

        # 2. GRU layer
        self.log_gru_layer(self.gru_2, layer_number=2)
        self.logger.info("---")
