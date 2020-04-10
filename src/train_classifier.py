import argparse
import logging
import os

import tensorflow as tf

from src.feature_extractor.extractor_factory import ExtractorFactory
from src.input_pipeline.dataset_factory import DatasetFactory
from src.input_pipeline.triplet_input_pipeline import TripletsInputPipeline
from src.models.classifier import Classifier
from src.models.model_factory import ModelFactory
from src.utils.params import Params
from src.utils.utils import Utils

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", default="experiments",
                    help="Experiment directory containing params.json")
parser.add_argument("--dataset_dir", default="DCASE",
                    help="Dataset directory containing the model")
parser.add_argument("--model_to_load", default="results/ConvNet1D-20200326-065709",
                    help="Model to load")

if __name__ == "__main__":
    # load the arguments
    args = parser.parse_args()

    # set the existing model to the experiment path
    experiment_path = os.path.join(args.experiment_dir, args.dataset_dir, args.model_to_load)
    # create folder for saving model
    saved_model_path = Utils.create_folder(os.path.join(experiment_path, "saved_model"))

    # load the params.json file from the existing model
    json_path = os.path.join(experiment_path, "logs", "params.json")
    params = Params(json_path)

    # define dataset
    dataset = DatasetFactory.create_dataset(name=params.dataset, params=params)
    # get the feature extractor from the factory
    extractor = ExtractorFactory.create_extractor(params.feature_extractor, params=params)
    # define triplet input pipeline
    pipeline = TripletsInputPipeline(params=params, dataset=dataset)

    # create model from factory and specified name within the params
    model = ModelFactory.create_model(params.model, embedding_dim=params.embedding_size)
    # create the optimizer for the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)

    # create the classifier model
    classifier = Classifier("Classifier", n_labels=len(dataset.LABELS))
    # loss for the classifier
    classifier_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # create the optimizer for the classifier
    classifier_optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)

    # create folders for experiment results
    experiment_name = "{0}-{1}".format(classifier.model_name, params.experiment_name)
    experiment_path, log_path, tensorb_path, save_path = Utils.create_load_folders_for_experiment(args,
                                                                                                  dataset_folder=dataset.EXPERIMENT_FOLDER,
                                                                                                  model_name=experiment_name)

    # set logger
    Utils.set_logger(log_path, params.log_level)
    logger = logging.getLogger("Main ({})".format(params.experiment_name))

    # set the folder for the summary writer
    train_summary_writer = tf.summary.create_file_writer(tensorb_path)

    # define triplet loss metrics
    classifier_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    classifier_accuracy_epochs = tf.keras.metrics.SparseCategoricalAccuracy()
    classifier_loss = tf.keras.metrics.Mean("classifier_accuracy", dtype=tf.float32)
    classifier_loss_epochs = tf.keras.metrics.Mean("classifier_accuracy_epochs", dtype=tf.float32)

    # define checkpoint and checkpoint manager
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, saved_model_path, max_to_keep=3)

    # check if models has been trained before
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        logger.info("Restored models from {}".format(manager.latest_checkpoint))
    else:
        logger.info("Initializing models from scratch.")

    for epoch in range(params.epochs):
        logger.info("Starting epoch {0} from {1}".format(epoch + 1, params.epochs))
        dataset_iterator = pipeline.get_dataset(extractor, shuffle=params.shuffle_dataset)
        dataset_iterator = iter(dataset_iterator)
        # iterate over the batches of the dataset
        for batch_index, (anchor, neighbour, opposite, triplet_labels) in enumerate(dataset_iterator):
            emb_anchor = model(anchor, training=False)
            emb_neighbour = model(neighbour, training=False)
            emb_opposite = model(opposite, training=False)

            # record the operations run during the forward pass, which enables auto differentiation
            with tf.GradientTape() as tape:
                # run the forward pass of the layer
                pred_anchor = classifier(emb_anchor)
                pred_neighbour = classifier(emb_neighbour)
                pred_opposite = classifier(emb_opposite)

                pred = tf.concat([pred_anchor, pred_neighbour, pred_opposite], axis=0)
                labels = tf.concat([triplet_labels[:, 0], triplet_labels[:, 1], triplet_labels[:, 2]], axis=0)

                c_loss = classifier_loss_fn(y_true=labels, y_pred=pred)

            # GradientTape automatically retrieves the gradients of the trainable variables with respect to the loss
            grads = tape.gradient(c_loss, classifier.trainable_weights)
            # update the values of the trainable variables to minimize the loss
            classifier_optimizer.apply_gradients(zip(grads, classifier.trainable_weights))

            # add losses to the metrics
            classifier_loss(c_loss)
            classifier_loss_epochs(c_loss)

            classifier_accuracy(labels, pred)
            classifier_accuracy_epochs(labels, pred)

            # write batch losses to summary writer
            with train_summary_writer.as_default():
                # write summary of batch losses
                tf.summary.scalar("classifier/loss_batches", classifier_loss.result(), step=int(ckpt.step))
                tf.summary.scalar("classifier/accuracy_batches", classifier_accuracy.result(), step=int(ckpt.step))

            logger.info("Classifier loss: {0}".format(c_loss))

            # add one step to checkpoint
            # todo - new ckpt for classifier
            ckpt.step.assign_add(1)

        # save the current model after a specified amount of epochs
        # TODO - saving classifier
        if epoch % params.save_frequency == 0 and bool(params.save_model):
            manager_save_path = manager.save()
            logger.info("Saved checkpoint for epoch {0}: {1}".format(epoch, manager_save_path))

        # write epoch loss to summary writer
        with train_summary_writer.as_default():
            # write summary of epoch loss
            tf.summary.scalar("classifier/loss_epochs", classifier_loss_epochs.result(), step=epoch)
            tf.summary.scalar("classifier/accuracy_epochs", classifier_accuracy_epochs.result(), step=int(ckpt.step))

        # reset metrics every epoch
        classifier_loss.reset_states()
        classifier_loss_epochs.reset_states()
