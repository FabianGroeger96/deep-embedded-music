import tensorflow as tf

from src.models.base_model import BaseModel
from src.models.model_factory import ModelFactory


@ModelFactory.register("DenseNet")
class DenseEncoder(BaseModel):
    """ A simple one layered dense model. """

    def __init__(self, embedding_dim, model_name="DenseEncoder"):
        """
        Initialises the model.
        Calls the initialise method of the super class.
        :param embedding_dim: the dimension for the embedding space.
        :param model_name: the name of the model.
        """
        super(DenseEncoder, self).__init__(embedding_dim=embedding_dim, model_name=model_name, expand_dims=False)

        self.dense = tf.keras.layers.Dense(embedding_dim, input_shape=(None, None, None), activation="relu")

    @tf.function
    def forward_pass(self, inputs):
        """
        The forward pass through the network.
        :param inputs: the input that will be passed through the model.
        :return: the output of the forward pass.
        """
        # Embedding layer
        features = self.flatten(inputs)
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
        pass
