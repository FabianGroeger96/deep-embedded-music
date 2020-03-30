import tensorflow as tf

from src.models.base_model import BaseModel
from src.models.model_factory import ModelFactory


@ModelFactory.register("GRUNet")
class GRUNet(BaseModel):
    def __init__(self, embedding_dim, model_name="GRUNet"):
        super(GRUNet, self).__init__(embedding_dim=embedding_dim, model_name=model_name, expand_dims=False)

        self.gru_1 = tf.keras.layers.GRU(64, return_sequences=True)
        self.gru_2 = tf.keras.layers.GRU(128)

    @tf.function
    def forward_pass(self, inputs):
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
