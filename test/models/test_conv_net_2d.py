import os

import tensorflow as tf

from src.feature_extractor.extractor_factory import ExtractorFactory
from src.dataset.dataset_factory import DatasetFactory
from src.input_pipeline.triplet_input_pipeline import TripletsInputPipeline
from src.loss.triplet_loss import TripletLoss
from src.models_embedding.conv_net_2d import ConvNet2D
from src.models_embedding.model_factory import ModelFactory
from src.training.train_model import train_step
from src.utils.params import Params
from src.utils.utils_audio import AudioUtils


class TestConvNet2D(tf.test.TestCase):

    def setUp(self):
        json_path = os.path.join("/tf/test_environment/", "config", "params.json")
        self.params = Params(json_path)

        self.audio_file_path = "/tf/test_environment/audio/DevNode1_ex1_1.wav"
        self.audio = AudioUtils.load_audio_from_file(self.audio_file_path,
                                                     sample_rate=16000,
                                                     sample_size=10,
                                                     stereo_channels=4,
                                                     to_mono=True)
        self.feature_extractor = ExtractorFactory.create_extractor("LogMelExtractor", params=self.params)
        self.audio_feature = self.feature_extractor.extract(self.audio)

        self.create_model()

        # create the optimizer for the model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate)
        # create the loss function for the model
        self.triplet_loss_fn = TripletLoss(margin=self.params.margin)

    def create_model(self):
        self.model = ModelFactory.create_model("ConvNet2D", embedding_dim=self.params.embedding_size,
                                               l2_amount=self.params.l2_amount)

    def get_input_pipeline(self):
        dataset = DatasetFactory.create_dataset("DCASEDataset", params=self.params)
        audio_pipeline = TripletsInputPipeline(params=self.params, dataset=dataset)

        return audio_pipeline

    def test_build_model(self):
        self.create_model()
        self.model.build(self.audio_feature.shape)

        self.assertDTypeEqual(self.model, ConvNet2D)

    def test_output_tensor(self):
        model_input = tf.expand_dims(self.audio_feature, axis=0)
        prediction = self.model(model_input)

        is_empty = tf.equal(tf.size(prediction), 0)
        self.assertFalse(is_empty)

    def test_loss_not_zero(self):
        self.create_model()
        # instantiate input pipeline
        audio_pipeline = self.get_input_pipeline()
        dataset_iterator = audio_pipeline.get_dataset(feature_extractor=self.feature_extractor)
        for anchor, neighbour, opposite, _ in dataset_iterator:
            batch = (anchor, neighbour, opposite)
            losses = train_step(batch, model=self.model, loss_fn=self.triplet_loss_fn, optimizer=self.optimizer)
            loss_triplet = losses["triplet_loss"]

            self.assertNotEqual(loss_triplet, 0)
            break


if __name__ == '__main__':
    tf.test.main()
