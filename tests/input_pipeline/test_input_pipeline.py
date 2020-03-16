import tensorflow as tf

from input_pipeline.dcase_data_frame import DCASEDataFrame
from input_pipeline.triplet_input_pipeline import TripletsInputPipeline


class TestInputPipeline(tf.test.TestCase):

    def test_iterator(self):
        audio_iterator = DCASEDataFrame(
            audio_files_path="/opt/project/tests/test_environment/",
            info_file_path="/opt/project/tests/test_environment/evaluation_setup/fold1_train.txt",
            sample_rate=16000)

        for audio, label, session, node_id, segment in audio_iterator:
            self.assertTrue(len(audio) > 0)
            self.assertNotEqual(label, "")
            self.assertNotEqual(session, "")
            self.assertNotEqual(node_id, "")
            self.assertNotEqual(segment, "")

    def test_dataset(self):
        audio_pipeline = TripletsInputPipeline(
            audio_files_path="/opt/project/tests/test_environment/",
            info_file_path="/opt/project/tests/test_environment/evaluation_setup/fold1_train.txt",
            sample_rate=16000,
            sample_size=10,
            batch_size=2,
            prefetch_batches=1,
            input_processing_buffer_size=1,
            category_cardinality=9)

        for triplet in audio_pipeline.get_dataset():
            print("Triplet:")
            print(triplet)

            anchors, neighbours, opposites, labels = triplet
            print("Anchor: {}".format(anchors))
            print("Neighbour: {}".format(neighbours))
            print("Opposite: {}".format(opposites))
            print("Labels: {}".format(labels))

            # check if there are the same number of samples as requested (batch_size)
            self.assertEqual(2, anchors.shape[0])
            self.assertEqual(2, neighbours.shape[0])
            self.assertEqual(2, opposites.shape[0])
            self.assertEqual(2, labels.shape[0])

            # check if triplets have the correct audio size (sample_rate * sample_size)
            self.assertEqual(16000 * 10, anchors.shape[1])
            self.assertEqual(16000 * 10, neighbours.shape[1])
            self.assertEqual(16000 * 10, opposites.shape[1])

            # check if triplets have a third dimension
            self.assertEqual(1, anchors.shape[2])
            self.assertEqual(1, neighbours.shape[2])
            self.assertEqual(1, opposites.shape[2])

            # check if triplets are valid
            # check if anchor and neighbour have the same labels
            self.assertTrue(labels[0][0] == labels[0][1])
            # check if anchor and opposite have different labels
            self.assertTrue(labels[0][0] != labels[0][2])


if __name__ == '__main__':
    tf.test.main()
