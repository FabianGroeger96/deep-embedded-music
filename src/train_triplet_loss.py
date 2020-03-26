import argparse
import io
import logging
import os

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.feature_extractor.log_mel_extractor import LogMelExtractor
from src.input_pipeline.dcase_data_frame import DCASEDataFrame
from src.input_pipeline.triplet_input_pipeline import TripletsInputPipeline
from src.loss.triplet_loss import TripletLoss
from src.models.ConvNet_1D import ConvNet1D
from src.models.ConvNet_2D import ConvNet2D
from src.models.Dense_Encoder import DenseEncoder
from src.models.ModelFactory import ModelFactory
from src.train_model import train_step
from src.utils.params import Params
from src.utils.utils import Utils
from src.utils.visualise_model import visualise_embeddings

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", default="experiments/DCASE",
                    help="Experiment directory containing params.json")

if __name__ == "__main__":
    # load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.experiment_dir, "config", "params.json")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # create model from factory and specified name within the params
    model = ModelFactory.create_model("ConvNet2D", embedding_dim=params.embedding_size)
    # create the optimizer for the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
    # create the loss function for the model
    triplet_loss_fn = TripletLoss(margin=params.margin)

    # create folders for experiment results
    experiment_path, log_path, tensorb_path, save_path = Utils.create_folders_for_experiment(args, model.model_name)

    # set logger
    Utils.set_logger(log_path, params.log_level)
    logger = logging.getLogger("Main ({})".format(params.experiment_name))

    # set the folder for the summary writer
    train_summary_writer = tf.summary.create_file_writer(tensorb_path)

    # define triplet loss metrics
    train_loss_triplet = tf.keras.metrics.Mean("train_loss_triplet", dtype=tf.float32)
    train_loss_neighbour = tf.keras.metrics.Mean("train_loss_neighbour", dtype=tf.float32)
    train_loss_opposite = tf.keras.metrics.Mean("train_loss_opposite", dtype=tf.float32)

    # define checkpoint and checkpoint manager
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, save_path, max_to_keep=3)

    # define triplet input pipeline
    pipeline = TripletsInputPipeline(
        dataset_path=params.audio_files_path,
        fold=params.fold,
        sample_rate=params.sample_rate,
        sample_size=params.sample_size,
        batch_size=params.batch_size,
        prefetch_batches=tf.data.experimental.AUTOTUNE,
        random_selection_buffer_size=params.random_selection_buffer_size,
        stereo_channels=params.stereo_channels,
        to_mono=params.to_mono,
        train_test_split_distribution=params.train_test_split)

    # define feature extractor
    extractor = LogMelExtractor(sample_rate=params.sample_rate,
                                sample_size=params.sample_size,
                                frame_length=params.frame_length,
                                frame_step=params.frame_step,
                                fft_size=params.fft_size,
                                n_mel_bin=params.n_mel_bin)

    # get test set for embedding visualisation
    test_features, test_labels = pipeline.get_test_dataset(extractor)

    # check if models has been trained before
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        logger.info("Restored models from {}".format(manager.latest_checkpoint))
    else:
        logger.info("Initializing models from scratch.")

    print_model = True

    # start of the training loop
    for epoch in range(1, params.epochs):
        logger.info("Starting epoch {0} from {1}".format(epoch, params.epochs))
        dataset_iterator = pipeline.get_dataset(extractor, shuffle=params.shuffle_dataset, calc_dist=params.calc_dist)
        dataset_iterator = iter(dataset_iterator)
        # iterate over the batches of the dataset
        # for batch_index, (anchor, neighbour, opposite, triplet_labels) in enumerate(dataset_iterator):
        #     if print_model:
        #         model.build(anchor.shape)
        #         model.summary(print_fn=logger.info)
        #         print_model = False
        #
        #     # run one training step
        #     batch = (anchor, neighbour, opposite, triplet_labels)
        #     losses = train_step(batch, model=model, loss_fn=triplet_loss_fn, optimizer=optimizer)
        #     loss_triplet, loss_neighbour, loss_opposite = losses
        #
        #     # add losses to the metrics
        #     train_loss_triplet(loss_triplet)
        #     train_loss_neighbour(loss_neighbour)
        #     train_loss_opposite(loss_opposite)
        #
        #     # log the current loss value of the batch
        #     logger.debug("Triplet loss at batch {0}: {1:1.2f}".format(batch_index + 1, float(loss_triplet)))
        #
        #     # write loss to summary writer
        #     with train_summary_writer.as_default():
        #         # write summary of losses
        #         tf.summary.scalar("triplet_loss/loss_triplet", train_loss_triplet.result(), step=int(ckpt.step))
        #         tf.summary.scalar("triplet_loss/loss_neighbour", train_loss_neighbour.result(), step=int(ckpt.step))
        #         tf.summary.scalar("triplet_loss/loss_opposite", train_loss_opposite.result(), step=int(ckpt.step))
        #
        #     if int(ckpt.step) % params.save_frequency == 0 and bool(params.save_model):
        #         # save the model
        #         manager_save_path = manager.save()
        #         logger.info("Saved checkpoint for step {}: {}".format(int(ckpt.step), manager_save_path))
        #
        #         # run test features through model
        #         embedding = model(test_features, training=False)
        #         # visualise test embeddings
        #         visualise_embeddings(embedding, test_labels, tensorb_path)
        #
        #     # log every 200 batches
        #     if int(ckpt.step) % 200 == 0:
        #         template = "Epoch {0}, Batch: {1}, Samples Seen: {2}, Triplet Loss: {3:1.2f}"
        #         logger.info(template.format(epoch,
        #                                     batch_index + 1,
        #                                     (batch_index + 1) * params.batch_size,
        #                                     train_loss_triplet.result()))
        #
        #     # add one step to checkpoint
        #     ckpt.step.assign_add(1)

        # reset metrics every epoch
        train_loss_triplet.reset_states()

        # reinitialise pipeline after epoch
        pipeline.reinitialise()

        #########
        # start visualisation
        #########
        # dataset_iterator = pipeline.get_dataset(extractor, shuffle=False, calc_dist=False)
        # dataset_iterator = iter(dataset_iterator)
        # lists for visualisation
        embeddings = []
        labels = []
        # iterate over the batches of the dataset
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

            if i > 3:
                break

        embeddings = tf.stack(embeddings)
        labels = tf.stack(labels)

        emb_np = embeddings.numpy()
        labels_np = labels.numpy()

        embeddings_by_lables = []
        for i, label in enumerate(DCASEDataFrame.LABELS):
            embeddings_class = tf.math.reduce_mean(emb_np[np.nonzero(labels_np == i)], 0)
            embeddings_by_lables.append(embeddings_class)
        embeddings_by_lables = tf.stack(embeddings_by_lables)

        pair_dist = tfa.losses.triplet.metric_learning.pairwise_distance(embeddings_by_lables)
        con_mat_df = pd.DataFrame(pair_dist.numpy(),
                                  index=DCASEDataFrame.LABELS,
                                  columns=DCASEDataFrame.LABELS)

        figure = plt.figure(figsize=(8, 8))
        plt.imshow(con_mat_df)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')

        plt.close(figure)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)

        image = tf.expand_dims(image, 0)

        # Log the confusion matrix as an image summary.
        with train_summary_writer.as_default():
            tf.summary.image("Confusion Matrix", image, step=epoch)

        # visualise test embeddings
        visualise_embeddings(embeddings, labels, tensorb_path)
