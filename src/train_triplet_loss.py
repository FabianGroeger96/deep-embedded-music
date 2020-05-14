import argparse
import logging
import os
from enum import Enum

import tensorflow as tf

from src.dataset.base_dataset import DatasetType
from src.dataset.dataset_factory import DatasetFactory
from src.feature_extractor.extractor_factory import ExtractorFactory
from src.input_pipeline.triplet_input_pipeline import TripletsInputPipeline
from src.loss.triplet_loss import TripletLoss
from src.models_embedding.model_factory import ModelFactory
from src.training.train_model import train_step
from src.utils.params import Params
from src.utils.utils import Utils
from src.utils.utils_tensorflow import set_gpu_experimental_growth
from src.utils.utils_visualise import visualise_model_on_epoch_end, visualise_embedding_on_training_end

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", default="experiments",
                    help="Experiment directory containing params.json")


class LogLevel(Enum):
    DEBUG = 0
    INFO = 1


def log_step(logger, epoch, batch_index, batch_size, result, log_level: LogLevel):
    template = "Epoch {0}, Batch: {1}, Samples Seen: {2}, Loss: {3:1.2f}, Triplet Loss: {4:1.2f}, Regularization " \
               "Loss: {5:1.2f}"

    loss = result["loss"]
    loss_triplet = result["triplet_loss"]
    loss_regularization = result["regularization_loss"]

    if log_level == LogLevel.INFO:
        logger.info(template.format(epoch + 1,
                                    batch_index + 1,
                                    (batch_index + 1) * batch_size,
                                    loss,
                                    loss_triplet,
                                    loss_regularization))

    elif log_level == LogLevel.DEBUG:
        logger.debug(template.format(epoch + 1,
                                     batch_index + 1,
                                     (batch_index + 1) * batch_size,
                                     loss,
                                     loss_triplet,
                                     loss_regularization))


def main():
    set_gpu_experimental_growth()

    # load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.experiment_dir, "config", "params.json")
    params = Params(json_path)

    # define dataset
    dataset = DatasetFactory.create_dataset(name=params.dataset, params=params)
    # get the feature extractor from the factory
    if not params.feature_extractor == "None":
        extractor = ExtractorFactory.create_extractor(params.feature_extractor, params=params)
    else:
        extractor = None
    # define triplet input pipeline
    pipeline = TripletsInputPipeline(params=params, dataset=dataset, log=False)

    # create model from factory and specified name within the params
    model = ModelFactory.create_model(params.model, params=params)
    # create the optimizer for the model
    decay_lr = tf.keras.optimizers.schedules.ExponentialDecay(params.learning_rate, 7500, 0.95, staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=decay_lr)
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
    summary_writer = tf.summary.create_file_writer(tensorb_path)

    # define triplet loss metrics
    train_joint_loss_epochs = tf.keras.metrics.Mean("train_joint_loss_epochs", dtype=tf.float32)
    train_triplet_loss_epochs = tf.keras.metrics.Mean("train_triplet_loss_epochs", dtype=tf.float32)
    train_regularization_loss_epochs = tf.keras.metrics.Mean("train_regularization_loss_epochs", dtype=tf.float32)

    # define checkpoint and checkpoint manager
    ckpt = tf.train.Checkpoint(optimizer=optimizer, net=model, step=tf.Variable(1))
    manager = tf.train.CheckpointManager(ckpt, save_path, max_to_keep=3)

    # check if models_embedding has been trained before
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        logger.info("Restored models_embedding from {}".format(manager.latest_checkpoint))
        # calculate the overall trained epochs
        epochs_before = int(ckpt.save_counter) * params.save_frequency
    else:
        logger.info("Initializing models_embedding from scratch.")
        epochs_before = 0

    # log the current dataset information
    dataset.print_dataset_info()

    # log the current models_embedding architecture
    print_model = True
    model.log_model()

    profiling = True
    if params.use_profiler:
        # profile execution for one epoch
        logger.info("Starting profiler")
        tf.summary.trace_on(graph=True, profiler=True)

    # start of the training loop
    for epoch in range(params.epochs):
        epochs_overall = epoch + epochs_before
        logger.info("Starting epoch {0} from {1}".format(epochs_overall + 1, params.epochs + epochs_before))
        dataset = pipeline.get_dataset(extractor, dataset_type=DatasetType.TRAIN,
                                       shuffle=params.shuffle_dataset)
        # iterate over the batches of the dataset
        for batch_index, (anchor, neighbour, opposite, _) in enumerate(dataset):
            if print_model:
                model.build(anchor.shape)
                model.summary(print_fn=logger.info)
                print_model = False

            # run one training step
            batch = (anchor, neighbour, opposite)
            losses = train_step(batch, model=model, loss_fn=triplet_loss_fn, optimizer=optimizer)

            # retrieve losses from step
            loss = losses["loss"]
            loss_triplet = losses["triplet_loss"]
            loss_regularization = losses["regularization_loss"]
            dist_neighbour = losses["dist_neighbour"]
            dist_opposite = losses["dist_opposite"]

            # add losses to the metrics for epoch metrics
            train_joint_loss_epochs(loss)
            train_triplet_loss_epochs(loss_triplet)
            train_regularization_loss_epochs(loss_regularization)

            # write batch losses to summary writer
            with summary_writer.as_default():
                # write summary of batch losses
                tf.summary.scalar("triplet_loss_train/loss_joint_batches", loss, step=int(ckpt.step))
                tf.summary.scalar("triplet_loss_train/loss_triplet_batches", loss_triplet, step=int(ckpt.step))
                tf.summary.scalar("triplet_loss_train/loss_regularization_batches", loss_regularization,
                                  step=int(ckpt.step))
                tf.summary.scalar("triplet_loss_train/dist_sq_neighbour", dist_neighbour, step=int(ckpt.step))
                tf.summary.scalar("triplet_loss_train/dist_sq_opposite", dist_opposite, step=int(ckpt.step))

            if int(ckpt.step) % 10 == 0 and params.use_profiler and profiling:
                logger.info("Stopping profiler at batch: {}".format(int(ckpt.step)))
                profiling = False
                with summary_writer.as_default():
                    tf.summary.trace_export(name="training_loop", step=int(ckpt.step), profiler_outdir=tensorb_path)

            # log the current loss value of the batch
            if int(ckpt.step) % 500 == 0:
                log_step(logger, epochs_overall, batch_index=batch_index, batch_size=params.batch_size,
                         result=losses, log_level=LogLevel.INFO)
            else:
                log_step(logger, epochs_overall, batch_index=batch_index, batch_size=params.batch_size,
                         result=losses, log_level=LogLevel.DEBUG)

            # add one step to checkpoint
            ckpt.step.assign_add(1)

        # save the current model after a specified amount of epochs
        if epoch % params.save_frequency == 0 and bool(params.save_model):
            manager_save_path = manager.save()
            logger.info("Saved checkpoint for epoch {0}: {1}".format(epochs_overall + 1, manager_save_path))

        # write epoch loss to summary writer
        with summary_writer.as_default():
            # write summary of epoch loss
            tf.summary.scalar("triplet_loss_train/loss_joint_epochs", train_joint_loss_epochs.result(),
                              step=epochs_overall)
            tf.summary.scalar("triplet_loss_train/loss_triplet_epochs", train_triplet_loss_epochs.result(),
                              step=epochs_overall)
            tf.summary.scalar("triplet_loss_train/loss_regularization_epochs",
                              train_regularization_loss_epochs.result(),
                              step=epochs_overall)

        # reset metrics every epoch
        train_joint_loss_epochs.reset_states()
        train_triplet_loss_epochs.reset_states()
        train_regularization_loss_epochs.reset_states()

        # visualise model on the end of a epoch, visualise embeddings, distance matrix, distance graphs
        visualise_model_on_epoch_end(model, pipeline=pipeline, extractor=extractor, epoch=epochs_overall,
                                     loss_fn=triplet_loss_fn, summary_writer=summary_writer,
                                     tensorb_path=tensorb_path)

        # reinitialise pipeline after epoch
        pipeline.reinitialise()
        logger.info("Epoch ended")

    # visualise the entire embedding space after finishing training
    visualise_embedding_on_training_end(model, pipeline=pipeline, extractor=extractor, tensorb_path=tensorb_path)


if __name__ == "__main__":
    main()
