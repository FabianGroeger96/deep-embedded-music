import argparse
import logging
import os
from enum import Enum

import tensorflow as tf

from src.feature_extractor.log_mel_extractor import LogMelExtractor
from src.input_pipeline.triplet_input_pipeline import TripletsInputPipeline
from src.loss.triplet_loss import TripletLoss
from src.models.model_factory import ModelFactory
from src.train_model import train_step
from src.utils.params import Params
from src.utils.utils import Utils
from src.utils.utils_visualise import visualise_model_on_epoch_end

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", default="experiments/DCASE",
                    help="Experiment directory containing params.json")


class LogLevel(Enum):
    DEBUG = 0
    INFO = 1


def log_step(epoch, batch_index, batch_size, result, log_level: LogLevel):
    template = "Epoch {0}, Batch: {1}, Samples Seen: {2}, Triplet Loss: {3:1.2f}"
    if log_level == LogLevel.INFO:
        logger.info(template.format(epoch + 1,
                                    batch_index + 1,
                                    (batch_index + 1) * batch_size,
                                    result))
    elif log_level == LogLevel.DEBUG:
        logger.debug(template.format(epoch + 1,
                                     batch_index + 1,
                                     (batch_index + 1) * batch_size,
                                     result))


if __name__ == "__main__":
    # load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.experiment_dir, "config", "params.json")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # create model from factory and specified name within the params
    model = ModelFactory.create_model(params.model, embedding_dim=params.embedding_size)
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
    train_loss_triplet_batches = tf.keras.metrics.Mean("train_loss_triplet_batches", dtype=tf.float32)
    train_loss_triplet_epochs = tf.keras.metrics.Mean("train_loss_triplet_epochs", dtype=tf.float32)
    train_loss_neighbour = tf.keras.metrics.Mean("train_loss_neighbour", dtype=tf.float32)
    train_loss_opposite = tf.keras.metrics.Mean("train_loss_opposite", dtype=tf.float32)

    # define checkpoint and checkpoint manager
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, save_path, max_to_keep=3)

    # define triplet input pipeline
    pipeline = TripletsInputPipeline(params=params)

    # define feature extractor
    extractor = LogMelExtractor(params=params)

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
    for epoch in range(params.epochs):
        logger.info("Starting epoch {0} from {1}".format(epoch + 1, params.epochs))
        dataset_iterator = pipeline.get_dataset(extractor, shuffle=params.shuffle_dataset, calc_dist=params.calc_dist)
        dataset_iterator = iter(dataset_iterator)
        # iterate over the batches of the dataset
        for batch_index, (anchor, neighbour, opposite, triplet_labels) in enumerate(dataset_iterator):
            if print_model:
                model.build(anchor.shape)
                model.summary(print_fn=logger.info)
                print_model = False

            # run one training step
            batch = (anchor, neighbour, opposite, triplet_labels)
            losses = train_step(batch, model=model, loss_fn=triplet_loss_fn, optimizer=optimizer)
            loss_triplet, loss_neighbour, loss_opposite = losses

            # add losses to the metrics
            train_loss_triplet_batches(loss_triplet)
            train_loss_triplet_epochs(loss_triplet)
            train_loss_neighbour(loss_neighbour)
            train_loss_opposite(loss_opposite)

            # write batch losses to summary writer
            with train_summary_writer.as_default():
                # write summary of batch losses
                tf.summary.scalar("triplet_loss/loss_triplet_batches", train_loss_triplet_batches.result(),
                                  step=int(ckpt.step))
                tf.summary.scalar("triplet_loss/loss_neighbour", train_loss_neighbour.result(), step=int(ckpt.step))
                tf.summary.scalar("triplet_loss/loss_opposite", train_loss_opposite.result(), step=int(ckpt.step))

            # log the current loss value of the batch
            if int(ckpt.step) % 500 == 0:
                log_step(epoch, batch_index=batch_index, batch_size=params.batch_size,
                         result=train_loss_triplet_batches.result(), log_level=LogLevel.INFO)
            else:
                log_step(epoch, batch_index=batch_index, batch_size=params.batch_size,
                         result=train_loss_triplet_batches.result(), log_level=LogLevel.DEBUG)

            # add one step to checkpoint
            ckpt.step.assign_add(1)

        # save the current model after a specified amount of epochs
        if epoch % params.save_frequency == 0 and bool(params.save_model):
            manager_save_path = manager.save()
            logger.info("Saved checkpoint for epoch {0}: {1}".format(epoch, manager_save_path))

        # write epoch loss to summary writer
        with train_summary_writer.as_default():
            # write summary of epoch loss
            tf.summary.scalar("triplet_loss/loss_triplet_epochs", train_loss_triplet_epochs.result(), step=epoch)

        # reset metrics every epoch
        train_loss_triplet_batches.reset_states()
        train_loss_triplet_epochs.reset_states()
        train_loss_neighbour.reset_states()
        train_loss_opposite.reset_states()

        # visualise model on the end of a epoch, visualise embeddings, distance matrix, distance graphs
        visualise_model_on_epoch_end(model, pipeline=pipeline, extractor=extractor, epoch=epoch,
                                     summary_writer=train_summary_writer, tensorboard_path=tensorb_path)

        # reinitialise pipeline after epoch
        pipeline.reinitialise()
