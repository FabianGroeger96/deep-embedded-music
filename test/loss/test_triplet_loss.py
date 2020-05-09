import os

import tensorflow as tf

from src.feature_extractor.extractor_factory import ExtractorFactory
from src.dataset.dataset_factory import DatasetFactory
from src.input_pipeline.triplet_input_pipeline import TripletsInputPipeline
from src.loss.triplet_loss import TripletLoss
from src.models_embedding.model_factory import ModelFactory
from src.utils.params import Params


class TestTripletLoss(tf.test.TestCase):

    def setUp(self):
        json_path = os.path.join("/tf/test_environment/", "config", "params.json")
        self.params = Params(json_path)

        self.feature_extractor = ExtractorFactory.create_extractor("LogMelExtractor", params=self.params)

        self.model = ModelFactory.create_model("ConvNet1D", embedding_dim=self.params.embedding_size)

        # create the optimizer for the model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate)
        # create the loss function for the model
        self.triplet_loss_fn = TripletLoss(margin=self.params.margin)

    def get_input_pipeline(self):
        dataset = DatasetFactory.create_dataset("DCASEDataset", params=self.params)
        audio_pipeline = TripletsInputPipeline(params=self.params, dataset=dataset)

        return audio_pipeline

    def test_triplet_loss(self):
        self.model = ModelFactory.create_model("ConvNet1D", embedding_dim=self.params.embedding_size)
        # instantiate input pipeline
        audio_pipeline = self.get_input_pipeline()
        dataset_iterator = audio_pipeline.get_dataset(feature_extractor=self.feature_extractor)
        for anchor, neighbour, opposite, triplet_labels in dataset_iterator:
            emb_anchor = self.model(anchor, training=True)
            emb_neighbour = self.model(neighbour, training=True)
            emb_opposite = self.model(opposite, training=True)

            # compute the triplet loss value for the batch
            triplet_loss = self.triplet_loss_fn(triplet_labels, [emb_anchor, emb_neighbour, emb_opposite])

            self.assertNotEqual(triplet_loss, 0)

            break

    def test_triplet_loss_distance(self):
        self.model = ModelFactory.create_model("ConvNet1D", embedding_dim=self.params.embedding_size)
        # instantiate input pipeline
        audio_pipeline = self.get_input_pipeline()
        dataset_iterator = audio_pipeline.get_dataset(feature_extractor=self.feature_extractor)
        for anchor, neighbour, opposite, triplet_labels in dataset_iterator:
            emb_anchor = self.model(anchor, training=True)
            emb_neighbour = self.model(neighbour, training=True)
            emb_opposite = self.model(opposite, training=True)

            # compute the distance losses between the embeddings
            dist_neighbour = self.triplet_loss_fn.calculate_distance(anchor=emb_anchor, embedding=emb_neighbour)
            dist_opposite = self.triplet_loss_fn.calculate_distance(anchor=emb_anchor, embedding=emb_opposite)

            self.assertNotEqual(dist_neighbour, 0)
            self.assertNotEqual(dist_opposite, 0)

            break


if __name__ == '__main__':
    tf.test.main()
