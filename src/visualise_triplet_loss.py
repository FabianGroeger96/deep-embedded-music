import argparse
import logging
import os

import tensorflow as tf

from src.feature_extractor.log_mel_extractor import LogMelExtractor
from src.input_pipeline.triplet_input_pipeline import TripletsInputPipeline
from src.loss.triplet_loss import TripletLoss
from src.models.model_factory import ModelFactory
from src.utils.params import Params
from src.utils.utils import Utils
from src.utils.utils_visualise import visualise_model_on_epoch_end

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
    model = ModelFactory.create_model(params.model, embedding_dim=params.embedding_size)
    # create the optimizer for the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
    # create the loss function for the model
    triplet_loss_fn = TripletLoss(margin=params.margin)

    # create folders for experiment results
    experiment_path, log_path, tensorb_path, save_path = Utils.create_folders_for_experiment(args, model.model_name,
                                                                                             saved_model_path=params.saved_model_path)

    # set logger
    Utils.set_logger(log_path, params.log_level)
    logger = logging.getLogger("Main ({})".format(params.experiment_name))

    # set the folder for the summary writer
    train_summary_writer = tf.summary.create_file_writer(tensorb_path)

    # define triplet input pipeline
    pipeline = TripletsInputPipeline(params=params)

    # define feature extractor
    extractor = LogMelExtractor(params=params)

    # define checkpoint and checkpoint manager
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, save_path, max_to_keep=3)

    # check if models has been trained before
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        logger.info("Restored models from {}".format(manager.latest_checkpoint))
    else:
        logger.info("Initializing models from scratch.")

    # visualise model on the end of a epoch, visualise embeddings, distance matrix, distance graphs
    visualise_model_on_epoch_end(model, pipeline=pipeline, extractor=extractor, epoch=99999,
                                 summary_writer=train_summary_writer, tensorboard_path=tensorb_path)
