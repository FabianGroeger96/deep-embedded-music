import os

import tensorflow as tf

from src.feature_extractor.extractor_factory import ExtractorFactory
from src.input_pipeline.dataset_factory import DatasetFactory
from src.input_pipeline.triplet_input_pipeline import TripletsInputPipeline
from src.loss.triplet_loss import TripletLoss
from src.models.model_factory import ModelFactory
from src.training.train_model import train_step
from src.utils.params import Params
from src.utils.utils_audio import AudioUtils


class TestGRUNet(tf.test.TestCase):

    def setUp(self):
        json_path = os.path.join("/opt/project/test_environment/", "config", "params.json")
        self.params = Params(json_path)

        self.audio_file_path = "/opt/project/test_environment/audio/DevNode1_ex1_1.wav"
        self.audio = AudioUtils.load_audio_from_file(self.audio_file_path,
                                                     sample_rate=16000,
                                                     sample_size=10,
                                                     stereo_channels=4,
                                                     to_mono=True)
        self.feature_extractor = ExtractorFactory.create_extractor("LogMelExtractor", params=self.params)
        self.audio_feature = self.feature_extractor.extract(self.audio)

        self.model = ModelFactory.create_model("GRUNet", embedding_dim=self.params.embedding_size)

        # create the optimizer for the model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate)
        # create the loss function for the model
        self.triplet_loss_fn = TripletLoss(margin=self.params.margin)

    def get_input_pipeline(self):
        dataset = DatasetFactory.create_dataset("DCASEDataset", params=self.params)
        audio_pipeline = TripletsInputPipeline(params=self.params, dataset=dataset)

        return audio_pipeline

    def test_output_tensor(self):
        model_input = tf.expand_dims(self.audio_feature, axis=0)
        prediction = self.model(model_input)

        is_empty = tf.equal(tf.size(prediction), 0)
        self.assertFalse(is_empty)

    def test_loss_not_zero(self):
        self.model = ModelFactory.create_model("GRUNet", embedding_dim=self.params.embedding_size)
        # instantiate input pipeline
        audio_pipeline = self.get_input_pipeline()
        dataset_iterator = audio_pipeline.get_dataset(feature_extractor=self.feature_extractor)
        for anchor, neighbour, opposite, triplet_labels in dataset_iterator:
            batch = (anchor, neighbour, opposite, triplet_labels)
            losses = train_step(batch, model=self.model, loss_fn=self.triplet_loss_fn, optimizer=self.optimizer)
            loss_triplet, dist_neighbour, dist_opposite = losses

            self.assertNotEqual(loss_triplet, 0)

            break

    def test_loss_decreases(self):
        self.model = ModelFactory.create_model("GRUNet", embedding_dim=self.params.embedding_size)

        loss_vals = []

        # instantiate input pipeline
        audio_pipeline = self.get_input_pipeline()
        for epoch in range(50):
            dataset_iterator = audio_pipeline.get_dataset(feature_extractor=self.feature_extractor)
            for anchor, neighbour, opposite, triplet_labels in dataset_iterator:
                batch = (anchor, neighbour, opposite, triplet_labels)
                losses = train_step(batch, model=self.model, loss_fn=self.triplet_loss_fn, optimizer=self.optimizer)
                loss_triplet, dist_neighbour, dist_opposite = losses
                loss_vals.append(loss_triplet)
                break

            audio_pipeline.reinitialise()

        self.assertLess(loss_vals[-1], loss_vals[0])


if __name__ == '__main__':
    tf.test.main()
