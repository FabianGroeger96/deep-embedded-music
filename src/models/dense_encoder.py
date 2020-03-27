import tensorflow as tf

from src.models.base_model import BaseModel
from src.models.model_factory import ModelFactory


@ModelFactory.register("DenseEncoder")
class DenseEncoder(BaseModel):
    def __init__(self, embedding_dim, model_name="DenseEncoder"):
        super(DenseEncoder, self).__init__(embedding_dim=embedding_dim, model_name=model_name, expand_dims=False)

        self.dense = tf.keras.layers.Dense(embedding_dim, input_shape=(None, None, None), activation="relu")

    @tf.function
    def forward_pass(self, inputs):
        # Embedding layer
        features = self.flatten(inputs)
        features = self.dense(features)
        # L2 normalisation
        features = self.l2_normalisation(features)

        return features
