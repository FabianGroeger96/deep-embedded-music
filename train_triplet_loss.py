import argparse
import logging
import os
from datetime import datetime

import tensorflow as tf
from tqdm import tqdm

from feature_extractor.log_mel_extractor import LogMelExtractor
from input_pipeline.triplet_input_pipeline import TripletsInputPipeline
from model.Dense_Encoder import DenseEncoder
from model.loss.triplet_loss import TripletLoss
from utils.params import Params
from utils.utils import set_logger

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", default="experiments/DCASE",
                    help="Experiment directory containing params.json")

if __name__ == "__main__":
    # load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.experiment_dir, "config", "params.json")
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # create experiment name folder
    experiment_path = os.path.join(args.experiment_dir, "results")
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    # Instantiate model, optimizer, triplet loss function
    model = DenseEncoder(embedding_dim=params.embedding_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
    triplet_loss_fn = TripletLoss(margin=params.margin)

    # create experiment time folder
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_folder_name = "{0}-{1}".format(model.model_name, current_time)
    experiment_path = os.path.join(experiment_path, experiment_folder_name)
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    # set logger
    set_logger(experiment_path, params.log_level)
    logger = logging.getLogger("Main ({})".format(params.experiment_name))

    # define checkpoint and checkpoint manager
    ckpt_path = os.path.join(experiment_path, "saved_model")
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=5)

    # set the folder for the summary writer
    train_log_dir = os.path.join(experiment_path, "train")
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # define triplet loss metric
    train_triplet_loss = tf.keras.metrics.Mean("train_triplet_loss", dtype=tf.float32)

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

    # check if model has been trained before
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        logger.info("Restored model from {}".format(manager.latest_checkpoint))
    else:
        logger.info("Initializing model from scratch.")

    # start of the training loop
    for epoch in tqdm(range(params.epochs), desc="Epochs"):
        logger.info("Starting epoch {0} from {1}".format(epoch, params.epochs))
        dataset_iterator = pipeline.get_dataset(extractor, shuffle=params.shuffle_dataset, calc_dist=params.calc_dist)
        # Iterate over the batches of the dataset.
        for anchor, neighbour, opposite, triplet_labels in dataset_iterator:
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto differentiation.
            with tf.GradientTape() as tape:
                # Run the forward pass of the layer.
                # The operations that the layer applies to its inputs are going to be recorded on the GradientTape.
                emb_anchor = model(anchor, training=True)
                emb_neighbour = model(neighbour, training=True)
                emb_opposite = model(opposite, training=True)

                # Compute the loss value for batch
                triplet_loss = triplet_loss_fn(triplet_labels, [emb_anchor, emb_neighbour, emb_opposite])
                logger.debug("Triplet loss at batch {0}: {1:1.2f}".format(int(ckpt.step), float(triplet_loss)))

            # Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect
            # to the loss.
            grads = tape.gradient(triplet_loss, model.trainable_weights)
            # Run one step of gradient descent by updating the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # add loss to the metric
            train_triplet_loss(triplet_loss)

            # Add one step to checkpoint
            ckpt.step.assign_add(1)

            # write loss to summary writer
            with train_summary_writer.as_default():
                tf.summary.scalar("triplet_loss", train_triplet_loss.result(), step=int(ckpt.step))

            # Log every 200 batches.
            if int(ckpt.step) % 200 == 0:
                template = "Epoch {0}, Batch: {1}, Samples Seen: {2}, Triplet Loss: {3:1.2f}"
                logger.info(template.format(epoch + 1,
                                            int(ckpt.step) + 1,
                                            (int(ckpt.step) + 1) * params.batch_size,
                                            train_triplet_loss.result()))

            # save the model every 10 steps
            if int(ckpt.step) % 10 == 0 and bool(params.save_model):
                save_path = manager.save()
                logger.info("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

        # reset metrics every epoch
        train_triplet_loss.reset_states()
