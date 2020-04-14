import argparse
import logging
import os
from enum import Enum

import tensorflow as tf

from src.feature_extractor.extractor_factory import ExtractorFactory
from src.input_pipeline.base_dataset import DatasetType
from src.input_pipeline.dataset_factory import DatasetFactory
from src.input_pipeline.triplet_input_pipeline import TripletsInputPipeline
from src.loss.triplet_loss import TripletLoss
from src.models.model_factory import ModelFactory
from src.train_model import train_step
from src.utils.params import Params
from src.utils.utils import Utils
from src.utils.utils_visualise import visualise_model_on_epoch_end

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", default="experiments",
                    help="Experiment directory containing params.json")


class LogLevel(Enum):
    DEBUG = 0
    INFO = 1


def log_step(logger, epoch, batch_index, batch_size, result, log_level: LogLevel):
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


def main():
    # load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.experiment_dir, "config", "params.json")
    params = Params(json_path)

    # define dataset
    dataset = DatasetFactory.create_dataset(name=params.dataset, params=params)
    # get the feature extractor from the factory
    extractor = ExtractorFactory.create_extractor(params.feature_extractor, params=params)
    # define triplet input pipeline
    pipeline = TripletsInputPipeline(params=params, dataset=dataset, log=True)

    # create model from factory and specified name within the params
    model = ModelFactory.create_model(params.model, embedding_dim=params.embedding_size)
    # create the optimizer for the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
    # create the loss function for the model
    triplet_loss_fn = TripletLoss(margin=params.margin)

    # create folders for experiment results
    experiment_name = "{0}-{1}".format(model.model_name, params.experiment_name)
    experiment_path, log_path, tensorb_path, save_path = Utils.create_load_folders_for_experiment(args,
                                                                                                  dataset_folder=dataset.EXPERIMENT_FOLDER,
                                                                                                  model_name=experiment_name,
                                                                                                  saved_model_path=params.saved_model_path)

    # set logger
    Utils.set_logger(log_path, params.log_level)
    logger = logging.getLogger("Main ({})".format(params.experiment_name))

    # print params
    params.print(json_path, logger=logger)

    # set the folder for the summary writer
    train_summary_writer = tf.summary.create_file_writer(tensorb_path)

    # define triplet loss metrics
    train_loss_triplet_batches = tf.keras.metrics.Mean("train_loss_triplet_batches", dtype=tf.float32)
    train_loss_triplet_epochs = tf.keras.metrics.Mean("train_loss_triplet_epochs", dtype=tf.float32)
    train_dist_neighbour = tf.keras.metrics.Mean("train_loss_neighbour", dtype=tf.float32)
    train_dist_opposite = tf.keras.metrics.Mean("train_loss_opposite", dtype=tf.float32)

    # define checkpoint and checkpoint manager
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, save_path, max_to_keep=3)

    # check if models has been trained before
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        logger.info("Restored models from {}".format(manager.latest_checkpoint))
    else:
        logger.info("Initializing models from scratch.")

    # log the current dataset information
    dataset.print_dataset_info()

    # log the current models architecture
    print_model = True
    model.log_model()

    profiling = True
    if params.use_profiler:
        # profile execution for one epoch
        logger.info("Starting profiler")
        tf.summary.trace_on(graph=True, profiler=True)

    # start of the training loop
    for epoch in range(params.epochs):
        logger.info("Starting epoch {0} from {1}".format(epoch + 1, params.epochs))
        dataset_iterator = pipeline.get_dataset(extractor, dataset_type=DatasetType.TRAIN,
                                                shuffle=params.shuffle_dataset)
        # iterate over the batches of the dataset
        for batch_index, (anchor, neighbour, opposite, _) in enumerate(dataset_iterator):
            if print_model:
                model.build(anchor.shape)
                model.summary(print_fn=logger.info)
                print_model = False

            # run one training step
            batch = (anchor, neighbour, opposite)
            losses = train_step(batch, model=model, loss_fn=triplet_loss_fn, optimizer=optimizer)
            loss_triplet, dist_neighbour, dist_opposite = losses

            # add losses to the metrics
            train_loss_triplet_batches(loss_triplet)
            train_loss_triplet_epochs(loss_triplet)
            train_dist_neighbour(dist_neighbour)
            train_dist_opposite(dist_opposite)

            # write batch losses to summary writer
            with train_summary_writer.as_default():
                # write summary of batch losses
                tf.summary.scalar("triplet_loss/loss_triplet_batches", train_loss_triplet_batches.result(),
                                  step=int(ckpt.step))
                tf.summary.scalar("triplet_loss/dist_sq_neighbour", train_dist_neighbour.result(), step=int(ckpt.step))
                tf.summary.scalar("triplet_loss/dist_sq_opposite", train_dist_opposite.result(), step=int(ckpt.step))

            if int(ckpt.step) % 10 == 0 and params.use_profiler and profiling:
                logger.info("Stopping profiler at batch: {}".format(int(ckpt.step)))
                profiling = False
                with train_summary_writer.as_default():
                    tf.summary.trace_export(name="training_loop", step=int(ckpt.step), profiler_outdir=tensorb_path)

            # log the current loss value of the batch
            if int(ckpt.step) % 500 == 0:
                log_step(logger, epoch, batch_index=batch_index, batch_size=params.batch_size,
                         result=train_loss_triplet_batches.result(), log_level=LogLevel.INFO)
            else:
                log_step(logger, epoch, batch_index=batch_index, batch_size=params.batch_size,
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
        train_dist_neighbour.reset_states()
        train_dist_opposite.reset_states()

        # visualise model on the end of a epoch, visualise embeddings, distance matrix, distance graphs
        visualise_model_on_epoch_end(model, pipeline=pipeline, extractor=extractor, epoch=epoch,
                                     summary_writer=train_summary_writer, tensorb_path=tensorb_path)

        # reinitialise pipeline after epoch
        pipeline.reinitialise()

        logger.info("Epoch ended")


if __name__ == "__main__":
    main()
