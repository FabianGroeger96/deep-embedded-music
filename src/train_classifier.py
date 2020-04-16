import argparse
import logging
import os

import tensorflow as tf
import tensorflow_addons as tfa

from src.feature_extractor.extractor_factory import ExtractorFactory
from src.input_pipeline.base_dataset import DatasetType
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
parser.add_argument("--model_to_load", default="results/ConvGRUNet-LogMel-l1e4-b32-ts2-ns4-m05-e256-20200414-110807",
                    help="Model to load")


def train():
    dataset_iterator = pipeline.get_dataset(extractor, dataset_type=DatasetType.TRAIN,
                                            shuffle=params.shuffle_dataset, return_labels=True)
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
            labels = tf.dtypes.cast(labels, tf.int32)

            loss = classifier_loss_fn(y_true=labels, y_pred=pred)

        # GradientTape automatically retrieves the gradients of the trainable variables with respect to the loss
        grads = tape.gradient(loss, classifier.trainable_weights)
        # update the values of the trainable variables to minimize the loss
        classifier_optimizer.apply_gradients(zip(grads, classifier.trainable_weights))

        metric_train_loss_batches(loss)
        metric_train_loss_epochs(loss)

        metric_train_accuracy_batches(labels, pred)
        metric_train_accuracy_epochs(labels, pred)

        metric_train_f1_batches.update_state(tf.one_hot(labels, len(dataset.LABELS)), pred)
        metric_train_f1_epochs.update_state(tf.one_hot(labels, len(dataset.LABELS)), pred)

        # write batch losses to summary writer
        with train_summary_writer.as_default():
            # write summary of batch losses
            tf.summary.scalar("classifier/train_loss_batches", metric_train_loss_batches.result(),
                              step=int(ckpt_classifier.step))
            tf.summary.scalar("classifier/train_accuracy_batches", metric_train_accuracy_batches.result(),
                              step=int(ckpt_classifier.step))
            tf.summary.scalar("classifier/train_f1_batches", metric_train_f1_batches.result(),
                              step=int(ckpt_classifier.step))

        logger.info("TRAIN - batch index: {0}, loss: {1:.2f}, acc: {2:.2f}, f1: {3:.2f}".format(batch_index, loss,
                                                                                                metric_train_accuracy_batches.result(),
                                                                                                metric_train_f1_batches.result()))

        if batch_index > 2:
            break

        # add one step to checkpoint
        ckpt_classifier.step.assign_add(1)

    # write epoch loss to summary writer
    with train_summary_writer.as_default():
        # write summary of epoch loss
        tf.summary.scalar("classifier/train_loss_epochs", metric_train_loss_epochs.result(), step=epoch)
        tf.summary.scalar("classifier/train_accuracy_epochs", metric_train_accuracy_epochs.result(), step=epoch)
        tf.summary.scalar("classifier/train_f1_epochs", metric_train_f1_epochs.result(), step=epoch)


def evaluate():
    logger.info("Starting to evaluate")
    dataset_iterator = pipeline.get_dataset(extractor, dataset_type=DatasetType.EVAL,
                                            shuffle=params.shuffle_dataset, return_labels=True)
    # iterate over the batches of the dataset
    for batch_index, (anchor, neighbour, opposite, triplet_labels) in enumerate(dataset_iterator):
        emb_anchor = model(anchor, training=False)
        emb_neighbour = model(neighbour, training=False)
        emb_opposite = model(opposite, training=False)

        # run the forward pass of the layer
        pred_anchor = classifier(emb_anchor)
        pred_neighbour = classifier(emb_neighbour)
        pred_opposite = classifier(emb_opposite)

        pred = tf.concat([pred_anchor, pred_neighbour, pred_opposite], axis=0)
        labels = tf.concat([triplet_labels[:, 0], triplet_labels[:, 1], triplet_labels[:, 2]], axis=0)
        labels = tf.dtypes.cast(labels, tf.int32)

        loss = classifier_loss_fn(y_true=labels, y_pred=pred)

        metric_eval_loss_epochs(loss)
        metric_eval_accuracy_epochs(labels, pred)
        metric_eval_f1_epochs.update_state(tf.one_hot(labels, len(dataset.LABELS)), pred)

        logger.info("EVAL - batch index: {0}, loss: {1:.2f}".format(batch_index, loss))

    # write epoch loss to summary writer
    with train_summary_writer.as_default():
        # write summary of epoch loss
        tf.summary.scalar("classifier/eval_loss_epochs", metric_eval_loss_epochs.result(), step=epoch)
        tf.summary.scalar("classifier/eval_accuracy_epochs", metric_eval_accuracy_epochs.result(), step=epoch)
        tf.summary.scalar("classifier/eval_f1_epochs", metric_eval_f1_epochs.result(), step=epoch)


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
    metric_train_accuracy_batches = tf.keras.metrics.SparseCategoricalAccuracy()
    metric_train_accuracy_epochs = tf.keras.metrics.SparseCategoricalAccuracy()

    metric_train_f1_batches = tfa.metrics.F1Score(num_classes=len(dataset.LABELS), average="macro")
    metric_train_f1_epochs = tfa.metrics.F1Score(num_classes=len(dataset.LABELS), average="macro")

    metric_train_loss_batches = tf.keras.metrics.Mean("train_loss_batches", dtype=tf.float32)
    metric_train_loss_epochs = tf.keras.metrics.Mean("train_loss_epochs", dtype=tf.float32)

    metric_eval_accuracy_epochs = tf.keras.metrics.SparseCategoricalAccuracy()
    metric_eval_f1_epochs = tfa.metrics.F1Score(num_classes=len(dataset.LABELS), average="macro")
    metric_eval_loss_epochs = tf.keras.metrics.Mean("eval_loss_epochs", dtype=tf.float32)

    # define checkpoint and checkpoint manager
    ckpt = tf.train.Checkpoint(net=model)
    manager = tf.train.CheckpointManager(ckpt, saved_model_path, max_to_keep=3)

    # check if models has been trained before
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        logger.info("Restored models from {}".format(manager.latest_checkpoint))
    else:
        logger.info("Initializing models from scratch.")

    ckpt_classifier = tf.train.Checkpoint(step=tf.Variable(1), net=classifier)

    for epoch in range(params.epochs):
        logger.info("Starting epoch {0} from {1}".format(epoch + 1, params.epochs))

        # train the classifier
        train()
        # evaluate the classifier
        evaluate()

        # reset metrics every epoch
        metric_train_accuracy_batches.reset_states()
        metric_train_accuracy_epochs.reset_states()
        metric_eval_accuracy_epochs.reset_states()

        metric_train_loss_batches.reset_states()
        metric_train_loss_epochs.reset_states()
        metric_eval_loss_epochs.reset_states()

        logger.info("Epoch end")
