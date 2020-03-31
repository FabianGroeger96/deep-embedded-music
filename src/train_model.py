import tensorflow as tf


# TODO fix bug "Check failed: IsAligned() ptr = 0x7f29de1276d0" when adding it as a tf.function
def train_step(batch, model, loss_fn, optimizer):
    """
    Runs one training step to train the model for one batch.

    :param batch: the batch to train the model on.
    :param model: the model to train.
    :param loss_fn: the loss function.
    :param optimizer: the optimizer for the training.
    :return: the value of the loss, distance to the neighbour and distance to the opposite.
    """
    anchor, neighbour, opposite, triplet_labels = batch

    # record the operations run during the forward pass, which enables auto differentiation
    with tf.GradientTape() as tape:
        # run the forward pass of the layer
        emb_anchor = model(anchor, training=True)
        emb_neighbour = model(neighbour, training=True)
        emb_opposite = model(opposite, training=True)

        # compute the triplet loss value for the batch
        triplet_loss = loss_fn(triplet_labels, [emb_anchor, emb_neighbour, emb_opposite])
        # compute the distance losses between the embeddings
        dist_neighbour = loss_fn.calculate_distance(anchor=emb_anchor, embedding=emb_neighbour)
        dist_opposite = loss_fn.calculate_distance(anchor=emb_anchor, embedding=emb_opposite)

    # GradientTape automatically retrieves the gradients of the trainable variables with respect to the loss
    grads = tape.gradient(triplet_loss, model.trainable_weights)
    # update the values of the trainable variables to minimize the loss
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return triplet_loss, dist_neighbour, dist_opposite


@tf.function
def predict_triplets(model, anchor, neighbour, opposite):
    """
    Predicts a triplet of features.

    :param model: the model used for predicting.
    :param anchor: the feature used as anchor.
    :param neighbour: the feature used as neighbour.
    :param opposite: the feature used as opposite.
    :return: the predicted embeddings of the anchor, neighbour and opposite feature.
    """
    emb_anchor = model(anchor, training=True)
    emb_neighbour = model(neighbour, training=True)
    emb_opposite = model(opposite, training=True)

    return emb_anchor, emb_neighbour, emb_opposite


@tf.function
def evaluation_step(feature, model):
    """
    Runs one evaluation step.

    :param feature: the feature to feed to the model.
    :param model: the model used for predicting.
    :return: the predicted embedding.
    """
    embedding = model(feature, training=False)

    return embedding
