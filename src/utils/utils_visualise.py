import io
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_addons as tfa
from tensorboard.plugins import projector
from sklearn import metrics

from src.dataset.base_dataset import DatasetType


def save_labels_tsv(labels, filename, log_dir, dataset):
    """
    Saves the labels of the features to a given path as a *.tsv file.
    The file can be used to visualise the embeddings within the projector.

    :param labels: the labels to save.
    :param filename: the name of the file.
    :param log_dir: the path of the file.
    :param dataset: the dataset of the input pipeline.
    :return: None.
    """
    with open(os.path.join(log_dir, filename), 'w') as f:
        for label in labels.numpy():
            f.write('{}\n'.format(dataset.LABELS[int(label)]))


def save_names_tsv(names, filename, log_dir):
    """
    Saves the names of the features to a given path as a *.tsv file.
    The file can be used to examine the embedding space.

    :param names: the labels to save.
    :param filename: the name of the file.
    :param log_dir: the path of the file.
    :param dataset: the dataset of the input pipeline.
    :return: None.
    """
    with open(os.path.join(log_dir, filename), 'w') as f:
        for name in names.numpy():
            f.write('{}\n'.format(str(name, encoding="utf-8")))


def save_embeddings_tsv(embeddings, filename, log_dir):
    """
    Saves the embeddings of the features to a given path as a *.tsv file.
    The file can be used to visualise the embeddings within the projector.

    :param embeddings: the embeddings to save.
    :param filename: the name of the file.
    :param log_dir: the path of the file.
    :return: None.
    """
    with open(os.path.join(log_dir, filename), 'w') as f:
        for embedding in embeddings.numpy():
            for vec in embedding:
                f.write('{}\t'.format(vec))
            f.write('\n')


def visualise_embeddings(embeddings, labels, names, dataset, tensorboard_path, save_checkpoint=True):
    """
    Visualises the embeddings with its corresponding labels with the projector of the tensorboard.

    :param embeddings: the embeddings to visualise.
    :param labels: the labels to visualise.
    :param names: then names of the segments they belong to.
    :param dataset: the dataset of the input pipeline.
    :param tensorboard_path: the path of the tensorboard files from the current experiment.
    :param save_checkpoint: if the embedding checkpoint should be saved or only the *.tsv files.
    :return: None.
    """
    # reshape the embedding to a flat tensor (batch_size, ?)
    # emb = tf.reshape(embeddings, [embeddings.shape[0], -1])
    tensor_embeddings = tf.Variable(embeddings, name='embeddings')

    # save labels and embeddings to .tsv, to assign each embedding a label
    save_labels_tsv(labels, 'labels.tsv', tensorboard_path, dataset=dataset)
    save_names_tsv(names, 'names.tsv', tensorboard_path)
    save_embeddings_tsv(embeddings, 'embeddings.tsv', tensorboard_path)

    # save the embeddings to a checkpoint file, which will then be loaded by the projector
    if save_checkpoint:
        saver = tf.compat.v1.train.Saver([tensor_embeddings])
        saver.save(sess=None, global_step=0, save_path=os.path.join(tensorboard_path, "embeddings.ckpt"))

        # register projector
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = "embeddings"
        embedding.metadata_path = "labels.tsv"
        projector.visualize_embeddings(tensorboard_path, config)


def save_graph(tensorboard_path, execute_callback, **args):
    """
    Saves the graph used to compute a tf.function, can be a model architecture or anything.

    :param tensorboard_path: the path of the tensorboard files from the current experiment.
    :param execute_callback: the function to visualise.
    :param args: the arguments which will be passed to the function.
    :return: the output of the given function call.
    """
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


def visualise_model_on_epoch_end(model, pipeline, extractor, epoch, loss_fn, summary_writer, tensorb_path,
                                 visualise_graphs=True, save_checkpoint=True):
    """
    Visualises the model at the end of an epoch on the evaluation dataset.

    It visualises the embedding space along with the projected embeddings.
    It visualises the distance between each label centroid as a distance matrix and as graphs.

    :param model: the model to visualise.
    :param pipeline: the input pipeline to provide the data.
    :param extractor: the extractor used to extract the features from the data.
    :param epoch: the current epoch.
    :param loss_fn: the triplet loss function, to calculate the loss on eval set.
    :param summary_writer: the summary writer of the tensorboard.
    :param tensorb_path: the path of the tensorboard files from the current experiment.
    :param visualise_graphs: if the graphs of the distances between clusters should be visualised.
    :param save_checkpoint: if the checkpoint of the embeddings should be saved.
    :return: None.
    """

    # reinitialise pipeline for visualisation
    pipeline.reinitialise()
    dataset_iterator = pipeline.get_dataset(extractor, shuffle=False, dataset_type=DatasetType.EVAL)

    # define triplet loss metrics
    metric_triplet_loss_epochs = tf.keras.metrics.Mean("eval_triplet_loss_epochs", dtype=tf.float32)
    metric_dist_neighbour = tf.keras.metrics.Mean("eval_dist_neighbour", dtype=tf.float32)
    metric_dist_opposite = tf.keras.metrics.Mean("eval_dist_opposite", dtype=tf.float32)

    # lists for embeddings and labels from entire dataset
    embeddings = []
    labels = []
    names = []

    # iterate over the batches of the dataset and feed batch to model
    for i, (anchor, neighbour, opposite, triplet_metadata) in enumerate(dataset_iterator):
        emb_anchor = model(anchor, training=False)
        emb_neighbour = model(neighbour, training=False)
        emb_opposite = model(opposite, training=False)

        # compute the triplet loss value for the batch
        triplet_loss = loss_fn(None, [emb_anchor, emb_neighbour, emb_opposite])
        # compute the distance losses between the embeddings
        dist_neighbour = loss_fn.calculate_distance(anchor=emb_anchor, embedding=emb_neighbour)
        dist_opposite = loss_fn.calculate_distance(anchor=emb_anchor, embedding=emb_opposite)

        # add losses to the metrics
        metric_triplet_loss_epochs(triplet_loss)
        metric_dist_neighbour(dist_neighbour)
        metric_dist_opposite(dist_opposite)

        embeddings.append(emb_anchor)
        embeddings.append(emb_neighbour)
        embeddings.append(emb_opposite)

        # retrieve labels from metadata
        triplet_labels = tf.strings.to_number(triplet_metadata[:, 0], tf.float32)
        triplet_names = triplet_metadata[:, 1]

        labels.append(triplet_labels[:, 0])
        labels.append(triplet_labels[:, 1])
        labels.append(triplet_labels[:, 2])

        names.append(triplet_names[:, 0])
        names.append(triplet_names[:, 1])
        names.append(triplet_names[:, 2])

    # stack the embeddings and labels to get a tensor from shape (dataset_size, ...)
    embeddings = tf.concat(embeddings, axis=0)
    labels = tf.concat(labels, axis=0)
    names = tf.concat(names, axis=0)

    # clustering metrics
    silhouette_score = metrics.silhouette_score(embeddings, labels, metric="euclidean")
    calinski_harabasz_score = metrics.calinski_harabasz_score(embeddings, labels)
    davies_bouldin_score = metrics.davies_bouldin_score(embeddings, labels)

    # write batch losses to summary writer
    with summary_writer.as_default():
        # write summary of batch losses
        tf.summary.scalar("triplet_loss_eval/loss_triplet_epochs", metric_triplet_loss_epochs.result(),
                          step=epoch)
        tf.summary.scalar("triplet_loss_eval/dist_sq_neighbour", metric_dist_neighbour.result(), step=epoch)
        tf.summary.scalar("triplet_loss_eval/dist_sq_opposite", metric_dist_opposite.result(), step=epoch)
        tf.summary.scalar("triplet_loss_eval/silhouette_score", silhouette_score, step=epoch)
        tf.summary.scalar("triplet_loss_eval/calinski_harabasz_score", calinski_harabasz_score, step=epoch)
        tf.summary.scalar("triplet_loss_eval/davies_bouldin_score", davies_bouldin_score, step=epoch)

    # get used dataset from pipeline
    dataset = pipeline.dataset

    # visualise the distance matrix with graph
    visualise_distance_matrix(embeddings, labels=labels, dataset=dataset, epoch=epoch, summary_writer=summary_writer,
                              visualise_graphs=visualise_graphs)
    # visualise embeddings from the entire dataset
    visualise_embeddings(embeddings, labels=labels, names=names, dataset=dataset, tensorboard_path=tensorb_path,
                         save_checkpoint=save_checkpoint)

    # delete unused lists of entire dataset
    del embeddings
    del labels


def visualise_embedding_on_training_end(model, pipeline, extractor, tensorb_path):
    """
    Visualises the embedding space at the end of the training over the entire dataset.

    :param model: the model to visualise.
    :param pipeline: the input pipeline to provide the data.
    :param extractor: the extractor used to extract the features from the data.
    :param tensorb_path: the path of the tensorboard files from the current experiment.
    :return: None.
    """
    pipeline.reinitialise()
    dataset_iterator = pipeline.get_dataset(extractor, shuffle=False, dataset_type=DatasetType.TRAIN_AND_EVAL)

    # lists for embeddings and labels from entire dataset
    embeddings = []
    labels = []
    names = []

    # iterate over the batches of the dataset and feed batch to model
    for batch_index, (anchor, neighbour, opposite, triplet_metadata) in enumerate(dataset_iterator):
        emb_anchor = model(anchor, training=False)
        emb_neighbour = model(neighbour, training=False)
        emb_opposite = model(opposite, training=False)

        embeddings.append(emb_anchor)
        embeddings.append(emb_neighbour)
        embeddings.append(emb_opposite)

        # retrieve labels from metadata
        triplet_labels = tf.strings.to_number(triplet_metadata[:, 0], tf.float32)
        triplet_names = triplet_metadata[:, 1]

        labels.append(triplet_labels[:, 0])
        labels.append(triplet_labels[:, 1])
        labels.append(triplet_labels[:, 2])

        names.append(triplet_names[:, 0])
        names.append(triplet_names[:, 1])
        names.append(triplet_names[:, 2])

    # stack the embeddings and labels to get a tensor from shape (dataset_size, ...)
    embeddings = tf.concat(embeddings, axis=0)
    labels = tf.concat(labels, axis=0)
    names = tf.concat(names, axis=0)

    # get used dataset from pipeline
    dataset = pipeline.dataset

    # visualise embeddings from the entire dataset
    visualise_embeddings(embeddings, labels=labels, names=names, dataset=dataset, tensorboard_path=tensorb_path,
                         save_checkpoint=True)

    # delete unused lists of entire dataset
    del embeddings
    del labels


def visualise_distance_matrix(embeddings, labels, dataset, epoch, summary_writer, visualise_graphs=True):
    """
    Visualise the distance matrix of given embeddings.

    It visualises the distance between each label centroid as a distance matrix and as graphs.

    :param embeddings: the embeddings to visualise.
    :param labels: the labels to visualise.
    :param dataset: the dataset of the input pipeline.
    :param epoch: the current epoch.
    :param summary_writer: the summary writer of the tensorboard.
    :param visualise_graphs: if the graphs of the distances between clusters should be visualised.
    :return: None.
    """
    emb_np = embeddings.numpy()
    labels_np = labels.numpy()

    # group the computed embeddings by labels
    embeddings_by_labels = []
    for i, label in enumerate(dataset.LABELS):
        embeddings_class = tf.math.reduce_mean(emb_np[np.nonzero(labels_np == i)], 0)
        embeddings_by_labels.append(embeddings_class)
    embeddings_by_labels = tf.stack(embeddings_by_labels)

    # compute the pairwise distance between the embeddings
    pair_dist = tfa.losses.triplet.metric_learning.pairwise_distance(embeddings_by_labels)
    # compute the confusion matrix from the distances between clusters
    distance_matrix = pd.DataFrame(pair_dist.numpy(),
                                   index=dataset.LABELS,
                                   columns=dataset.LABELS)

    # visualise the distance graphs
    if visualise_graphs:
        visualise_distance_graphs(distance_matrix, epoch=epoch, summary_writer=summary_writer)
    # visualise the distance matrix as an image
    visualise_distance_matrix_image(distance_matrix, dataset=dataset, epoch=epoch, summary_writer=summary_writer)

    # delete unused big lists
    del emb_np
    del labels_np


def visualise_distance_graphs(distance_matrix, epoch, summary_writer):
    """
    Visualise the distances between each label centroid as graphs.

    :param distance_matrix: the distance matrix of the label centroids.
    :param epoch: the current epoch.
    :param summary_writer: the summary writer of the tensorboard .
    :return: None.
    """
    # extract the lower triangle matrix from the confusion matrix and transform it to a pd df
    lower_triangle_dist = distance_matrix.where(
        np.triu(np.ones(distance_matrix.shape), k=1).astype(bool)).stack().reset_index()
    # write loss to summary writer
    with summary_writer.as_default():
        for index, row in lower_triangle_dist.iterrows():
            # write summary of distance between clusters
            summary_name = "distances_between_clusters/{0}_to_{1}".format(row["level_0"], row["level_1"])
            tf.summary.scalar(summary_name, row[0], step=epoch)


def visualise_distance_matrix_image(distance_matrix, dataset, epoch, summary_writer):
    """
    Visualise the distances between each label centroid as an image.

    :param distance_matrix: the distance matrix of the label centroids.
    :param dataset: the dataset of the input pipeline.
    :param epoch: the current epoch.
    :param summary_writer: the summary writer of the tensorboard .
    :return: None.
    """
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(distance_matrix, annot=True, xticklabels=dataset.LABELS, yticklabels=dataset.LABELS)
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
