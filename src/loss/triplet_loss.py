import tensorflow as tf


class TripletLoss(tf.keras.losses.Loss):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    @tf.function
    def calculate_distance(self, anchor, embedding):
        distance = tf.reduce_sum(tf.square(tf.subtract(anchor, embedding)), axis=-1)
        loss = tf.reduce_sum(distance)

        return loss

    @tf.function
    def call(self, y_true, y_pred):
        anchor = y_pred[0]
        neighbour = y_pred[1]
        opposite = y_pred[2]

        dist_neighbour = tf.reduce_sum(tf.square(tf.subtract(anchor, neighbour)), axis=-1)
        dist_opposite = tf.reduce_sum(tf.square(tf.subtract(anchor, opposite)), axis=-1)

        loss = tf.add(tf.subtract(dist_neighbour, dist_opposite), self.margin)
        loss = tf.maximum(0.0, loss)
        loss = tf.reduce_sum(loss)

        return loss
