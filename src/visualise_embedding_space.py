import argparse
import os

import tensorflow as tf

from src.dataset.dataset_factory import DatasetFactory
from src.feature_extractor.extractor_factory import ExtractorFactory
from src.input_pipeline.triplet_input_pipeline import TripletsInputPipeline
from src.models_embedding.model_factory import ModelFactory
from src.utils.params import Params
from src.utils.utils_visualise import visualise_embedding_on_training_end

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", default="experiments",
                    help="Experiment directory containing params.json")
parser.add_argument("--dataset_dir", default="DCASE",
                    help="Dataset directory containing the model")
parser.add_argument("--model_to_load",
                    default="results/ResNet18-LogMel-l1e5-d95-7500-b64-l2001-ts5-ns5-m1-e256-20200511-155329",
                    help="Model to load")

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # load the arguments
    args = parser.parse_args()

    # load the parameters of the model to train
    # set the existing model to the experiment path
    experiment_path = os.path.join(args.experiment_dir, args.dataset_dir, args.model_to_load)
    # load folder for saving model
    saved_model_path = os.path.join(experiment_path, "saved_model")
    # folder to save embeddings
    tensorb_path = os.path.join(experiment_path, "tensorboard")

    # get the parameters from the json file
    params_saved_model = Params(os.path.join(experiment_path, "logs", "params.json"))

    # load the embedding model
    model_embedding = ModelFactory.create_model(params_saved_model.model, params=params_saved_model)
    # define checkpoint and checkpoint manager
    ckpt = tf.train.Checkpoint(net=model_embedding)
    manager = tf.train.CheckpointManager(ckpt, saved_model_path, max_to_keep=3)

    # check if models_embedding has been trained before
    ckpt.restore(manager.latest_checkpoint)
    if not manager.latest_checkpoint:
        raise ValueError("Embedding model could not be restored")

    # define dataset
    dataset = DatasetFactory.create_dataset(name=params_saved_model.dataset, params=params_saved_model)
    # get the feature extractor from the factory
    extractor = ExtractorFactory.create_extractor(params_saved_model.feature_extractor, params=params_saved_model)
    # define triplet input pipeline
    pipeline = TripletsInputPipeline(params=params_saved_model, dataset=dataset)

    # visualise the entire embedding space after finishing training
    visualise_embedding_on_training_end(model_embedding, pipeline=pipeline, extractor=extractor,
                                        tensorb_path=tensorb_path)
