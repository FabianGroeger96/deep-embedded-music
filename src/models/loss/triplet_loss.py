import tensorflow as tf


class TripletLoss(tf.keras.losses.Loss):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def call(self, y_true, y_pred):
        anchor = y_pred[0]
        neighbour = y_pred[1]
        opposite = y_pred[2]

        neighbour_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, neighbour)), axis=-1)
        opposite_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, opposite)), axis=-1)

        loss = tf.add(tf.subtract(neighbour_dist, opposite_dist), self.margin)
        loss = tf.maximum(0.0, loss)
        loss = tf.reduce_sum(loss)

        return loss
