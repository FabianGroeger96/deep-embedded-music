import argparse
import logging
import os

import tensorflow as tf

from src.feature_extractor.log_mel_extractor import LogMelExtractor
from src.input_pipeline.triplet_input_pipeline import TripletsInputPipeline
from src.loss.triplet_loss import TripletLoss
from src.models.Dense_Encoder import DenseEncoder
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

    # instantiate models, optimizer, triplet loss function
    model = DenseEncoder(embedding_dim=params.embedding_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
    triplet_loss_fn = TripletLoss(margin=params.margin)

    # create folders for experiment results
    experiment_path, log_path, tensorb_path, save_path = Utils.create_folders_for_experiment(args, model.model_name)

    # set logger
    Utils.set_logger(log_path, params.log_level)
    logger = logging.getLogger("Main ({})".format(params.experiment_name))

    # set the folder for the summary writer
    train_summary_writer = tf.summary.create_file_writer(tensorb_path)

    # define triplet loss metric
    train_triplet_loss = tf.keras.metrics.Mean("train_triplet_loss", dtype=tf.float32)

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
        to_mono=params.to_mono)

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

    # start of the training loop
    for epoch in range(params.epochs):
        logger.info("Starting epoch {0} from {1}".format(epoch, params.epochs))
        dataset_iterator = pipeline.get_dataset(extractor, shuffle=params.shuffle_dataset, calc_dist=params.calc_dist)
        # iterate over the batches of the dataset
        for anchor, neighbour, opposite, triplet_labels in dataset_iterator:
            # trace the current graph
            tf.summary.trace_on(graph=True, profiler=False)

            # run one training step
            triplet_loss = train_step((anchor, neighbour, opposite, triplet_labels),
                                      model=model, loss_fn=triplet_loss_fn, optimizer=optimizer)
            # add loss to the metric
            train_triplet_loss(triplet_loss)

            # log the current loss value of the batch
            logger.debug("Triplet loss at batch {0}: {1:1.2f}".format(int(ckpt.step), float(triplet_loss)))

            # write loss to summary writer
            with train_summary_writer.as_default():
                # write summary of loss
                tf.summary.scalar("triplet_loss", train_triplet_loss.result(), step=int(ckpt.step))
                # write summary of the graph
                tf.summary.trace_export(name="model_trace", step=int(ckpt.step), profiler_outdir=tensorb_path)

            if int(ckpt.step) % 20 == 0 and bool(params.save_model):
                # save the model
                manager_save_path = manager.save()
                logger.info("Saved checkpoint for step {}: {}".format(int(ckpt.step), manager_save_path))

                # run test features through model
                embedding = model(test_features, training=False)
                # visualise test embeddings
                visualise_embeddings(embedding, test_labels, tensorb_path)

            # log every 200 batches
            if int(ckpt.step) % 200 == 0:
                template = "Epoch {0}, Batch: {1}, Samples Seen: {2}, Triplet Loss: {3:1.2f}"
                logger.info(template.format(epoch + 1,
                                            int(ckpt.step) + 1,
                                            (int(ckpt.step) + 1) * params.batch_size,
                                            train_triplet_loss.result()))

            # add one step to checkpoint
            ckpt.step.assign_add(1)

        # reset metrics every epoch
        train_triplet_loss.reset_states()
