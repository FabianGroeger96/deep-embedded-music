import tensorflow as tf


@tf.function
def train_step(batch, model, loss_fn, optimizer):
    anchor, neighbour, opposite, triplet_labels = batch

    # record the operations run during the forward pass, which enables auto differentiation
    with tf.GradientTape() as tape:
        # run the forward pass of the layer
        emb_anchor = model(anchor, training=True)
        emb_neighbour = model(neighbour, training=True)
        emb_opposite = model(opposite, training=True)

        # compute the triplet loss value for the batch
        triplet_loss = loss_fn(triplet_labels, [emb_anchor, emb_neighbour, emb_opposite])

    # GradientTape automatically retrieves the gradients of the trainable variables with respect to the loss
    grads = tape.gradient(triplet_loss, model.trainable_weights)
    # update the values of the trainable variables to minimize the loss
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return triplet_loss


@tf.function
def evaluation_step(audio, model):
    embedding = model(audio, training=False)

    return embedding