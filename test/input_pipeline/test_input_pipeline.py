import tensorflow as tf

from src.feature_extractor.log_mel_extractor import LogMelExtractor
from src.feature_extractor.mfcc_extractor import MFCCExtractor
from src.input_pipeline.dcase_data_frame import DCASEDataFrame
from src.input_pipeline.triplet_input_pipeline import TripletsInputPipeline


class TestInputPipeline(tf.test.TestCase):

    def setUp(self):
        self.dataset_path = "/opt/project/test/test_environment/"
        self.fold = 1
        self.sample_rate = 16000
        self.sample_size = 10
        self.batch_size = 2
        self.prefetch_batches = 2
        self.random_selection_buffer_size = 1
        self.stereo_channels = 4
        self.to_mono = False

        n_frame = 98
        fft_size = 512
        n_mel_bin = 128
        n_mfcc_bin = 13

    def get_input_pipeline(self):
        audio_pipeline = TripletsInputPipeline(
            dataset_path=self.dataset_path,
            fold=self.fold,
            sample_rate=self.sample_rate,
            sample_size=self.sample_size,
            batch_size=self.batch_size,
            prefetch_batches=self.prefetch_batches,
            random_selection_buffer_size=self.random_selection_buffer_size,
            stereo_channels=self.stereo_channels,
            to_mono=self.to_mono)
        return audio_pipeline

    def test_data_frame_iterator(self):
        audio_iterator = DCASEDataFrame(
            dataset_path=self.dataset_path,
            fold=self.fold,
            sample_rate=self.sample_rate)

        for audio_entry in audio_iterator:
            self.assertNotEqual(audio_entry.file_name, "")
            self.assertNotEqual(audio_entry.label, "")
            self.assertNotEqual(audio_entry.session, "")
            self.assertNotEqual(audio_entry.node_id, "")
            self.assertNotEqual(audio_entry.segment, "")

    def test_dataset_generator_channels(self):
        # single channel
        self.stereo_channels = 1
        self.to_mono = True
        audio_pipeline = self.get_input_pipeline()
        dataset_iterator = audio_pipeline.get_dataset(feature_extractor=None, shuffle=True,
                                                      calc_dist=False)
        for anchor, neighbour, opposite, triplet_labels in dataset_iterator:
            # check if triplets have a third dimension (channel)
            self.assertEqual([self.batch_size, self.sample_rate], anchor.shape)
            self.assertEqual([self.batch_size, self.sample_rate], neighbour.shape)
            self.assertEqual([self.batch_size, self.sample_rate], opposite.shape)

        # multiple channels
        self.stereo_channels = 4
        self.to_mono = False
        audio_pipeline = self.get_input_pipeline()
        dataset_iterator = audio_pipeline.get_dataset(feature_extractor=None, shuffle=True, calc_dist=False)
        for anchor, neighbour, opposite, triplet_labels in dataset_iterator:
            # check if triplets have a third dimension (channel)
            self.assertEqual([self.batch_size, self.sample_rate, self.stereo_channels], anchor.shape)
            self.assertEqual([self.batch_size, self.sample_rate, self.stereo_channels], neighbour.shape)
            self.assertEqual([self.batch_size, self.sample_rate, self.stereo_channels], opposite.shape)

    def test_dataset_generator_batch_size(self):
        audio_pipeline = self.get_input_pipeline()
        dataset_iterator = audio_pipeline.get_dataset(feature_extractor=None, shuffle=True, calc_dist=False)
        for anchor, neighbour, opposite, triplet_labels in dataset_iterator:
            # check if there are the same number of samples as requested (batch_size)
            self.assertEqual(self.batch_size, anchor.shape[0])
            self.assertEqual(self.batch_size, neighbour.shape[0])
            self.assertEqual(self.batch_size, opposite.shape[0])
            self.assertEqual(self.batch_size, triplet_labels.shape[0])

    def test_dataset_generator_sample_rate_audios(self):
        audio_pipeline = self.get_input_pipeline()
        dataset_iterator = audio_pipeline.get_dataset(feature_extractor=None, shuffle=True, calc_dist=False)
        for anchor, neighbour, opposite, triplet_labels in dataset_iterator:
            # check if triplets have the correct audio size (sample_rate)
            self.assertEqual(self.sample_rate, anchor.shape[1])
            self.assertEqual(self.sample_rate, neighbour.shape[1])
            self.assertEqual(self.sample_rate, opposite.shape[1])

    def test_dataset_generator_triplets_valid(self):
        audio_pipeline = self.get_input_pipeline()
        dataset_iterator = audio_pipeline.get_dataset(feature_extractor=None, shuffle=True, calc_dist=False)
        for anchor, neighbour, opposite, triplet_labels in dataset_iterator:
            # check if triplets are valid
            for a_label, n_label, o_label in triplet_labels:
                # check if anchor and neighbour have the same labels
                self.assertTrue(a_label == n_label)
                # check if anchor and opposite have different labels
                self.assertTrue(a_label != o_label)

    def test_dataset_generator_mfcc_extractor(self):
        feature_extractor = MFCCExtractor(sample_rate=self.sample_rate, sample_size=self.sample_size, frame_length=480,
                                          frame_step=160, fft_size=512, n_mel_bin=128, n_mfcc_bin=13)

        frame_size, n_mfcc_bin = feature_extractor.get_output_shape()

        audio_pipeline = self.get_input_pipeline()
        dataset_iterator = audio_pipeline.get_dataset(feature_extractor=feature_extractor, shuffle=True,
                                                      calc_dist=False)
        for anchor, neighbour, opposite, triplet_labels in dataset_iterator:
            # check if triplets have the correct shape
            self.assertEqual([self.batch_size, frame_size, n_mfcc_bin, self.stereo_channels],
                             anchor.shape)
            self.assertEqual([self.batch_size, frame_size, n_mfcc_bin, self.stereo_channels],
                             neighbour.shape)
            self.assertEqual([self.batch_size, frame_size, n_mfcc_bin, self.stereo_channels],
                             opposite.shape)

    def test_dataset_generator_log_mel_extractor(self):
        feature_extractor = LogMelExtractor(sample_rate=self.sample_rate, sample_size=self.sample_size,
                                            frame_length=480,
                                            frame_step=160, fft_size=512, n_mel_bin=128)

        frame_size, n_mel_bin = feature_extractor.get_output_shape()

        audio_pipeline = self.get_input_pipeline()
        dataset_iterator = audio_pipeline.get_dataset(feature_extractor=feature_extractor, shuffle=True,
                                                      calc_dist=False)
        for anchor, neighbour, opposite, triplet_labels in dataset_iterator:
            # check if triplets have the correct shape
            self.assertEqual([self.batch_size, frame_size, n_mel_bin, self.stereo_channels],
                             anchor.shape)
            self.assertEqual([self.batch_size, frame_size, n_mel_bin, self.stereo_channels],
                             neighbour.shape)
            self.assertEqual([self.batch_size, frame_size, n_mel_bin, self.stereo_channels],
                             opposite.shape)


if __name__ == '__main__':
    tf.test.main()
