import argparse
import logging
import os

import tensorflow as tf

from src.feature_extractor.extractor_factory import ExtractorFactory
from src.input_pipeline.triplet_input_pipeline import TripletsInputPipeline
from src.loss.triplet_loss import TripletLoss
from src.models.model_factory import ModelFactory
from src.utils.params import Params
from src.utils.utils import Utils
from src.utils.utils_visualise import visualise_model_on_epoch_end

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", default="experiments/DCASE",
                    help="Experiment directory containing params.json")
parser.add_argument("--model_to_load", default="results/ConvNet1D-20200326-065709",
                    help="Model to load")

if __name__ == "__main__":
    # load the arguments
    args = parser.parse_args()

    # create folders for experiment results
    experiment_path, log_path, tensorb_path, save_path = Utils.create_load_folders_for_experiment(args, model_name=None,
                                                                                                  saved_model_path=args.model_to_load,
                                                                                                  copy_json_file=False)

    # load the params.json file from the existing model
    json_path = os.path.join(experiment_path, "logs", "params.json")
    params = Params(json_path)

    # create model from factory and specified name within the params
    model = ModelFactory.create_model(params.model, embedding_dim=params.embedding_size)
    # create the optimizer for the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
    # create the loss function for the model
    triplet_loss_fn = TripletLoss(margin=params.margin)

    # set logger
    Utils.set_logger(log_path, params.log_level)
    logger = logging.getLogger("Main ({})".format(params.experiment_name))

    # set the folder for the summary writer
    train_summary_writer = tf.summary.create_file_writer(tensorb_path)

    # define triplet input pipeline
    pipeline = TripletsInputPipeline(params=params)

    # get the feature extractor from the factory
    extractor = ExtractorFactory.create_extractor(params.feature_extractor, params=params)

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
                                 summary_writer=train_summary_writer, tensorb_path=tensorb_path, reinitialise=False,
                                 visualise_graphs=False, save_checkpoint=False)
