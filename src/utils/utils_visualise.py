import io
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_addons as tfa
from tensorboard.plugins import projector

from src.input_pipeline.dcase_dataset import DCASEDataset


def save_labels_tsv(labels, filepath, log_dir):
    with open(os.path.join(log_dir, filepath), 'w') as f:
        for label in labels.numpy():
            f.write('{}\n'.format(DCASEDataset.LABELS[int(label)]))


def save_embeddings_tsv(embeddings, filepath, log_dir):
    with open(os.path.join(log_dir, filepath), 'w') as f:
        for embedding in embeddings.numpy():
            for vec in embedding:
                f.write('{}\t'.format(vec))
            f.write('\n')


def visualise_embeddings(embeddings, triplet_labels, tensorboard_path):
    # reshape the embedding to a flat tensor (batch_size, ?)
    # emb = tf.reshape(embeddings, [embeddings.shape[0], -1])
    tensor_embeddings = tf.Variable(embeddings, name='embeddings')

    # save labels and embeddings to .tsv, to assign each embedding a label
    save_labels_tsv(triplet_labels[0], 'labels.tsv', tensorboard_path)
    save_embeddings_tsv(embeddings, 'embeddings.tsv', tensorboard_path)

    # save the embeddings to a checkpoint file, which will then be loaded by the projector
    saver = tf.compat.v1.train.Saver([tensor_embeddings])
    saver.save(sess=None, global_step=0, save_path=os.path.join(tensorboard_path, "embeddings.ckpt"))

    # register projector
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "embeddings"
    embedding.metadata_path = "labels.tsv"
    projector.visualize_embeddings(tensorboard_path, config)


def save_graph(tensorboard_path, execute_callback, **args):
    writer = tf.summary.create_file_writer(tensorboard_path)

    # Bracket the function call with
    # tf.summary.trace_on() and tf.summary.trace_export().
    tf.summary.trace_on(graph=True, profiler=True)
    # Call only one tf.function when tracing.
    r = execute_callback(**args)

    with writer.as_default():
        tf.summary.trace_export(
            name="model_graph",
            step=0,
            profiler_outdir=tensorboard_path)

    return r


def visualise_model_on_epoch_end(model, pipeline, extractor, epoch, summary_writer, tensorboard_path):
    # reinitialise pipeline for visualisation
    pipeline.reinitialise()
    dataset_iterator = pipeline.get_dataset(extractor, shuffle=False, calc_dist=False)
    dataset_iterator = iter(dataset_iterator)

    # lists for embeddings and labels from entire dataset
    embeddings = []
    labels = []
    # iterate over the batches of the dataset and feed batch to model
    for i, (anchor, neighbour, opposite, triplet_labels) in enumerate(dataset_iterator):
        emb_anchor = model(anchor, training=False)
        emb_neighbour = model(neighbour, training=False)
        emb_opposite = model(opposite, training=False)

        embeddings.append(emb_anchor)
        embeddings.append(emb_neighbour)
        embeddings.append(emb_opposite)

        labels.append(triplet_labels[:, 0])
        labels.append(triplet_labels[:, 1])
        labels.append(triplet_labels[:, 2])

    # stack the embeddings and labels to get a tensor from shape (dataset_size, ...)
    embeddings = tf.stack(embeddings)
    labels = tf.stack(labels)

    # visualise the distance matrix with graph and confusion matrix
    visualise_distance_matrix(embeddings, labels, epoch, summary_writer)
    # visualise embeddings from the entire dataset
    visualise_embeddings(embeddings, labels, tensorboard_path)

    # delete unused lists of entire dataset
    del embeddings
    del labels


def visualise_distance_matrix(embeddings, labels, epoch, summary_writer):
    emb_np = embeddings.numpy()
    labels_np = labels.numpy()

    # group the computed embeddings by labels
    embeddings_by_lables = []
    for i, label in enumerate(DCASEDataset.LABELS):
        embeddings_class = tf.math.reduce_mean(emb_np[np.nonzero(labels_np == i)], 0)
        embeddings_by_lables.append(embeddings_class)
    embeddings_by_lables = tf.stack(embeddings_by_lables)

    # compute the pairwise distance between the embeddings
    pair_dist = tfa.losses.triplet.metric_learning.pairwise_distance(embeddings_by_lables)
    # compute the confusion matrix from the distances between clusters
    distance_matrix = pd.DataFrame(pair_dist.numpy(),
                                   index=DCASEDataset.LABELS,
                                   columns=DCASEDataset.LABELS)

    # visualise the distance graphs
    visualise_distance_graphs(distance_matrix, epoch, summary_writer)
    # visualise the distance matrix as an image
    visualise_distance_matrix_image(distance_matrix, epoch, summary_writer)

    # delete unused big lists
    del emb_np
    del labels_np


def visualise_distance_graphs(distance_matrix, epoch, summary_writer):
    # extract the lower triangle matrix from the confusion matrix and transform it to a pd df
    lower_triangle_dist = distance_matrix.where(
        np.triu(np.ones(distance_matrix.shape), k=1).astype(bool)).stack().reset_index()
    # write loss to summary writer
    with summary_writer.as_default():
        for index, row in lower_triangle_dist.iterrows():
            # write summary of distance between clusters
            summary_name = "distances_between_clusters/{0}_to_{1}".format(row["level_0"], row["level_1"])
            tf.summary.scalar(summary_name, row[0], step=epoch)


def visualise_distance_matrix_image(distance_matrix, epoch, summary_writer):
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(distance_matrix, annot=True, xticklabels=DCASEDataset.LABELS, yticklabels=DCASEDataset.LABELS)
    plt.tight_layout()
    plt.ylabel("Distance from label center")
    plt.xlabel("Distance to label center")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(figure)
    buf.seek(0)

    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    with summary_writer.as_default():
        tf.summary.image("Label center distance matrix", image, step=epoch)
