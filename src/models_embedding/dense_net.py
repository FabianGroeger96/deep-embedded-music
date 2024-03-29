import tensorflow as tf

from src.models_embedding.base_model import BaseModel
from src.models_embedding.model_factory import ModelFactory


@ModelFactory.register("DenseNet")
class DenseEncoder(BaseModel):
    """ A simple one layered dense model. """

    def __init__(self, params, model_name="DenseEncoder"):
        """
        Initialises the model.
        Calls the initialise method of the super class.

        :param params: the global hyperparameters for initialising the model.
        :param model_name: the name of the model.
        """
        super(DenseEncoder, self).__init__(params=params, model_name=model_name, expand_dims=False)

        self.dense = tf.keras.layers.Dense(params.embedding_size, input_shape=(None, None, None), activation="relu")

    @tf.function
    def forward_pass(self, inputs, training=None):
        """
        The forward pass through the network.

        :param inputs: the input that will be passed through the model.
        :param training: if the model is training, for disabling dropout, batch norm. etc.
        :return: the output of the forward pass.
        """
        # Embedding layer
        x = self.flatten(inputs)
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
        pass
