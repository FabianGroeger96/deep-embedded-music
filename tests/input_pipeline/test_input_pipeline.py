import tensorflow as tf

from feature_extractor.log_mel_extractor import LogMelExtractor
from input_pipeline.dcase_data_frame import DCASEDataFrame
from input_pipeline.triplet_input_pipeline import TripletsInputPipeline


class TestInputPipeline(tf.test.TestCase):

    def test_iterator(self):
        audio_iterator = DCASEDataFrame(
            dataset_path="/opt/project/tests/test_environment/",
            fold=1,
            sample_rate=16000)

        for audio_entry in audio_iterator:
            self.assertNotEqual(audio_entry.file_name, "")
            self.assertNotEqual(audio_entry.label, "")
            self.assertNotEqual(audio_entry.session, "")
            self.assertNotEqual(audio_entry.node_id, "")
            self.assertNotEqual(audio_entry.segment, "")

    def test_dataset(self):

        audio_pipeline = TripletsInputPipeline(
            dataset_path="/opt/project/tests/test_environment/",
            fold=1,
            sample_rate=16000,
            sample_size=10,
            batch_size=2,
            prefetch_batches=2,
            random_selection_buffer_size=1)

        n_frame = 98
        fft_size = 512
        n_mel_bin = 128
        n_mfcc_bin = 13
        fingerprint_size = n_frame * n_mfcc_bin  # frame_size * mfcc_bin_size

        feature_extractor = LogMelExtractor(sample_rate=16000, fft_size=512, n_mel_bin=128)

        for anchor, neighbour, opposite, triplet_labels in audio_pipeline.get_dataset(feature_extractor,
                                                                                      shuffle=True,
                                                                                      calc_dist=False):

            # check if there are the same number of samples as requested (batch_size)
            self.assertEqual(2, anchor.shape[0])
            self.assertEqual(2, neighbour.shape[0])
            self.assertEqual(2, opposite.shape[0])
            self.assertEqual(2, triplet_labels.shape[0])

            # # check if triplets have the correct audio size (sample_rate * sample_size)
            # self.assertEqual(16000 * 10, anchor.shape[0])
            # self.assertEqual(16000 * 10, neighbour.shape[0])
            # self.assertEqual(16000 * 10, opposite.shape[0])
            #
            # # check if triplets have a third dimension
            # self.assertEqual(1, anchor.shape[1])
            # self.assertEqual(1, neighbour.shape[1])
            # self.assertEqual(1, opposite.shape[1])

            # check if triplets are valid
            for a_label, n_label, o_label in triplet_labels:
                # check if anchor and neighbour have the same labels
                self.assertTrue(a_label == n_label)
                # check if anchor and opposite have different labels
                self.assertTrue(a_label != o_label)


if __name__ == '__main__':
    tf.test.main()
