import argparse
import logging
import os
from datetime import datetime

import tensorflow as tf

from feature_extractor.log_mel_extractor import LogMelExtractor
from input_pipeline.triplet_input_pipeline import TripletsInputPipeline
from model.Dense_Encoder import DenseEncoder
from model.loss.triplet_loss import TripletLoss
from utils.params import Params
from utils.utils import set_logger

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="experiments/triplet_loss",
                    help="Experiment directory containing params.json")

if __name__ == "__main__":
    # load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, "params.json")
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # create experiment name folder
    experiment_path = os.path.join(args.model_dir, params.experiment_name)
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    # create experiment time folder
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_path = os.path.join(experiment_path, current_time)
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    # set logger
    set_logger(experiment_path, params.log_level)
    logger = logging.getLogger("Main ({})".format(params.experiment_name))

    # set the folder for the summary writer
    train_log_dir = os.path.join(experiment_path, "train")
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # define triplet loss metric
    train_triplet_loss = tf.keras.metrics.Mean("train_triplet_loss", dtype=tf.float32)

    # Instantiate model, optimizer, triplet loss function
    model = DenseEncoder(embedding_dim=params.embedding_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
    triplet_loss_fn = TripletLoss(margin=params.margin)

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

    extractor = LogMelExtractor(sample_rate=params.sample_rate,
                                sample_size=params.sample_size,
                                frame_length=params.frame_length,
                                frame_step=params.frame_step,
                                fft_size=params.fft_size,
                                n_mel_bin=params.n_mel_bin)

    for epoch in range(params.epochs):
        logger.info("Starting epoch {0} from {1}".format(epoch, params.epochs))
        dataset_iterator = pipeline.get_dataset(extractor, shuffle=params.shuffle_dataset, calc_dist=params.calc_dist)
        # Iterate over the batches of the dataset.
        for step, (anchor, neighbour, opposite, triplet_labels) in enumerate(dataset_iterator):
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
                logger.debug("Triplet loss at batch {0}: {1}".format(step, float(triplet_loss)))

            # Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect
            # to the loss.
            grads = tape.gradient(triplet_loss, model.trainable_weights)
            # Run one step of gradient descent by updating the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # add loss to the metric
            train_triplet_loss(triplet_loss)

            # write loss to summary writer
            with train_summary_writer.as_default():
                tf.summary.scalar("triplet_loss", train_triplet_loss.result(), step=step)

            # Log every 200 batches.
            if step % 200 == 0:
                template = "Epoch {0}, Batch: {1}, Samples Seen: {2}, Triplet Loss: {3}"
                logger.info(template.format(epoch + 1,
                                            step + 1,
                                            (step + 1) * params.batch_size,
                                            train_triplet_loss.result()))

        # reset metrics every epoch
        train_triplet_loss.reset_states()
