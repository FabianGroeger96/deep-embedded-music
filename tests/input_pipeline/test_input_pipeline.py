import tensorflow as tf

from input_pipeline.dcase_data_frame import DCASEDataFrame
from input_pipeline.triplet_input_pipeline import TripletsInputPipeline


class TestInputPipeline(tf.test.TestCase):

    def test_iterator(self):
        audio_iterator = DCASEDataFrame(
            audio_files_path="/opt/project/datasets/DCASE18-Task5-development/",
            info_file_path="/opt/project/datasets/DCASE18-Task5-development/evaluation_setup/fold1_train.txt",
            sample_rate=16000)

        for audio, label, session, node_id, segment in audio_iterator:
            self.assertTrue(len(audio) > 0)
            self.assertNotEqual(label, "")
            self.assertNotEqual(session, "")
            self.assertNotEqual(node_id, "")
            self.assertNotEqual(segment, "")

    def test_dataset(self):
        audio_pipeline = TripletsInputPipeline(
            audio_files_path="/opt/project/datasets/DCASE18-Task5-development/",
            info_file_path="/opt/project/datasets/DCASE18-Task5-development/evaluation_setup/fold1_train.txt",
            sample_rate=16000,
            sample_size=10,
            batch_size=2,
            prefetch_batches=1,
            input_processing_buffer_size=1)

        for triplet in audio_pipeline.get_dataset():
            print("Triplet:")
            print(triplet)

            anchor, neighbour, opposite = triplet
            print("Anchor: {}".format(anchor))
            print("Neighbour: {}".format(neighbour))
            print("Opposite: {}".format(opposite))

            # check if there are the same number of samples as requested (batch_size)
            self.assertEqual(2, anchor.shape[0])
            self.assertEqual(2, neighbour.shape[0])
            self.assertEqual(2, opposite.shape[0])

            # check if triplets have the correct audio size (sample_rate * sample_size)
            self.assertEqual(16000 * 10, anchor.shape[1])
            self.assertEqual(16000 * 10, neighbour.shape[1])
            self.assertEqual(16000 * 10, opposite.shape[1])

            # check if triplets have a third dimension
            self.assertEqual(1, anchor.shape[2])
            self.assertEqual(1, neighbour.shape[2])
            self.assertEqual(1, opposite.shape[2])
