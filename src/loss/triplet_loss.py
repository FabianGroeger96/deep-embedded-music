import tensorflow as tf


class TripletLoss(tf.keras.losses.Loss):
    """
    Implementation of the triplet loss.
    Special kind of loss function which gets three embeddings as an input; an anchor, neighbour and opposite.
    The anchor and the neighbour are of the same class and the anchor and the opposite have two different classes.
    The goal of this loss is to minimize the distance between the anchor and the neighbour, and to maximize the
    distance between the anchor and the opposite."""

    def __init__(self, margin):
        """
        Initialises the triplet loss.

        :param margin: the margin between the distance of the anchor to the neighbour and the anchor to the opposite.
        """
        super(TripletLoss, self).__init__()
        self.margin = margin

    @tf.function
    def calculate_distance(self, anchor, embedding):
        """
        Calculates the distance between the anchor and an embedding.

        :param anchor: the embedding of the anchor.
        :param embedding: the embedding to calculate the distance.
        :return: the distance between the embeddings.
        """
        distance = tf.reduce_sum(tf.square(tf.subtract(anchor, embedding)), axis=-1)
        distance_sum = tf.reduce_sum(distance)

        return distance_sum

    @tf.function
    def call(self, y_true, y_pred):
        """
        Calculates the triplet loss of the given embeddings.

        :param y_true: the embeddings to calculate the triplet loss from. 3 items within the array.
                        Structure: [0] anchor, [1] neighbour, [2] opposite.
        :param y_pred: the corresponding labels to the embeddings. 3 items within the array.
                        Structure: [0] anchor label, [1] neighbour label, [2] opposite label
        :return: the triplet loss.
        """
        anchor = y_pred[0]
        neighbour = y_pred[1]
        opposite = y_pred[2]

        dist_neighbour = tf.reduce_sum(tf.square(tf.subtract(anchor, neighbour)), axis=-1)
        dist_opposite = tf.reduce_sum(tf.square(tf.subtract(anchor, opposite)), axis=-1)

        loss = tf.add(tf.subtract(dist_neighbour, dist_opposite), self.margin)
        loss = tf.maximum(0.0, loss)
        loss = tf.reduce_mean(loss)

        return loss
