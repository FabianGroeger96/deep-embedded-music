import argparse
import logging
import os

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from sklearn.metrics import classification_report

from src.dataset.base_dataset import DatasetType
from src.dataset.dataset_factory import DatasetFactory
from src.feature_extractor.extractor_factory import ExtractorFactory
from src.input_pipeline.triplet_input_pipeline import TripletsInputPipeline
from src.models_classifier.classifier_dense import ClassifierDense
from src.models_classifier.classifier_logistic import ClassifierLogistic
from src.models_embedding.model_factory import ModelFactory
from src.utils.params import Params
from src.utils.utils import Utils
from src.utils.utils_tensorflow import set_gpu_experimental_growth

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", default="experiments",
                    help="Experiment directory containing params.json")
parser.add_argument("--dataset_dir", default="DCASE",
                    help="Dataset directory containing the model")
parser.add_argument("--model_to_load",
                    default="results/ResNet18-LogMel-l1e5-b128-l2001-d95-7500-ts5-ns5-m1-e128-20200511-162832",
                    help="Model to load")
parser.add_argument("--classifier_type",
                    default="Logistic",
                    help="Which classifier to use, dense or logistic")
parser.add_argument("--eval_on_test_at_end",
                    default="True",
                    help="If the embedding model an the on top trained classifier should be evaluated on the test set.")


def training():
    dataset_iterator = pipeline.get_dataset(extractor, dataset_type=DatasetType.TRAIN,
                                            shuffle=params_classifier.shuffle_dataset, return_labels=True)
    # iterate over the batches of the dataset
    for batch_index, (anchor, neighbour, opposite, triplet_metadata) in enumerate(dataset_iterator):
        # embed the triplets into the embedding space
        emb_anchor, emb_neighbour, emb_opposite = embed_triplet(anchor, neighbour, opposite)

        # retrieve labels from metadata
        triplet_labels = tf.strings.to_number(triplet_metadata[:, 0], tf.float32)

        # record the operations run during the forward pass, which enables auto differentiation
        with tf.GradientTape() as tape:
            # run the forward pass of the layer
            pred_anchor = classifier(emb_anchor)

            #pred_neighbour = classifier(emb_neighbour)
            #pred_opposite = classifier(emb_opposite)

            #pred = tf.concat([pred_anchor, pred_neighbour, pred_opposite], axis=0)
            #labels = tf.concat([triplet_labels[:, 0], triplet_labels[:, 1], triplet_labels[:, 2]], axis=0)
            #labels = tf.dtypes.cast(labels, tf.int32)

            pred = tf.concat([pred_anchor], axis=0)
            labels = tf.concat([triplet_labels[:, 0]], axis=0)
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

        logger.info("TRAIN - epoch: {0}, batch index: {1}, loss: {2:.2f}, acc: {3:.2f}, f1: {4:.2f}".format(
            epoch + 1,
            batch_index,
            metric_train_loss_batches.result(),
            metric_train_accuracy_batches.result(),
            metric_train_f1_batches.result()))

        # reset metrics every epoch
        metric_train_accuracy_batches.reset_states()
        metric_train_loss_batches.reset_states()
        metric_train_f1_batches.reset_states()

        # add one step to checkpoint
        ckpt_classifier.step.assign_add(1)

    # write epoch loss to summary writer
    with train_summary_writer.as_default():
        # write summary of epoch loss
        tf.summary.scalar("classifier/train_loss_epochs", metric_train_loss_epochs.result(), step=epoch)
        tf.summary.scalar("classifier/train_accuracy_epochs", metric_train_accuracy_epochs.result(), step=epoch)
        tf.summary.scalar("classifier/train_f1_epochs", metric_train_f1_epochs.result(), step=epoch)


def evaluating():
    logger.info("Starting to evaluate")
    pipeline.reinitialise()
    dataset_iterator = pipeline.get_dataset(extractor, dataset_type=DatasetType.EVAL,
                                            shuffle=params_classifier.shuffle_dataset, return_labels=True)

    # arrays to save the labels and predictions over the entire eval set
    global_labels = []
    global_predictions = []

    # iterate over the batches of the dataset
    for batch_index, batch in enumerate(dataset_iterator):
        # embed the triplets into the embedding space
        #pred, labels = get_predictions_from_triplet(batch)

        anchor, neighbour, opposite, triplet_metadata = batch
        # embed the triplets into the embedding space
        emb_anchor, emb_neighbour, emb_opposite = embed_triplet(anchor, neighbour, opposite)

        # retrieve labels from metadata
        triplet_labels = tf.strings.to_number(triplet_metadata[:, 0], tf.float32)

        # run the forward pass of the layer
        pred_anchor = classifier(emb_anchor)

        # pred_neighbour = classifier(emb_neighbour)
        # pred_opposite = classifier(emb_opposite)

        # pred = tf.concat([pred_anchor, pred_neighbour, pred_opposite], axis=0)
        # labels = tf.concat([triplet_labels[:, 0], triplet_labels[:, 1], triplet_labels[:, 2]], axis=0)
        # labels = tf.dtypes.cast(labels, tf.int32)

        pred = tf.concat([pred_anchor], axis=0)
        labels = tf.concat([triplet_labels[:, 0]], axis=0)
        labels = tf.dtypes.cast(labels, tf.int32)

        global_labels.append(labels)
        global_predictions.append(np.argmax(pred.numpy(), axis=1))

        loss = classifier_loss_fn(y_true=labels, y_pred=pred)

        metric_eval_loss_epochs(loss)
        metric_eval_accuracy_epochs(labels, pred)
        metric_eval_f1_epochs.update_state(tf.one_hot(labels, len(dataset.LABELS)), pred)

        logger.info("EVAL - epoch: {0}, batch index: {1}, loss: {2:.2f}".format(epoch + 1, batch_index, loss))

    # write epoch loss to summary writer
    with train_summary_writer.as_default():
        # write summary of epoch loss
        tf.summary.scalar("classifier/eval_loss_epochs", metric_eval_loss_epochs.result(), step=epoch)
        tf.summary.scalar("classifier/eval_accuracy_epochs", metric_eval_accuracy_epochs.result(), step=epoch)
        tf.summary.scalar("classifier/eval_f1_epochs", metric_eval_f1_epochs.result(), step=epoch)

    global_labels = tf.concat(global_labels, axis=0)
    global_predictions = tf.concat(global_predictions, axis=0)
    class_report = classification_report(global_labels, global_predictions, target_names=dataset.LABELS)
    logger.info("Classification Report:\n{}".format(class_report))


def testing():
    logger.info("Starting to test")
    pipeline.reinitialise()
    dataset_iterator = pipeline.get_dataset(extractor, dataset_type=DatasetType.TEST,
                                            shuffle=params_classifier.shuffle_dataset, return_labels=True)

    # arrays to save the labels and predictions over the entire eval set
    global_labels = []
    global_predictions = []

    # iterate over the batches of the dataset
    for batch_index, batch in enumerate(dataset_iterator):
        # embed the triplets into the embedding space
        # pred, labels = get_predictions_from_triplet(batch)

        anchor, neighbour, opposite, triplet_metadata = batch
        # embed the triplets into the embedding space
        emb_anchor, emb_neighbour, emb_opposite = embed_triplet(anchor, neighbour, opposite)

        # retrieve labels from metadata
        triplet_labels = tf.strings.to_number(triplet_metadata[:, 0], tf.float32)

        # run the forward pass of the layer
        pred_anchor = classifier(emb_anchor)

        # pred_neighbour = classifier(emb_neighbour)
        # pred_opposite = classifier(emb_opposite)

        # pred = tf.concat([pred_anchor, pred_neighbour, pred_opposite], axis=0)
        # labels = tf.concat([triplet_labels[:, 0], triplet_labels[:, 1], triplet_labels[:, 2]], axis=0)
        # labels = tf.dtypes.cast(labels, tf.int32)

        pred = tf.concat([pred_anchor], axis=0)
        labels = tf.concat([triplet_labels[:, 0]], axis=0)
        labels = tf.dtypes.cast(labels, tf.int32)

        global_labels.append(labels)
        global_predictions.append(np.argmax(pred.numpy(), axis=1))

        metric_test_accuracy_epochs(labels, pred)
        metric_test_f1_epochs.update_state(tf.one_hot(labels, len(dataset.LABELS)), pred)

        logger.info("TEST - batch index: {0}, f1: {1:.2f}".format(batch_index, metric_test_f1_epochs.result()))

    # write epoch loss to summary writer
    with train_summary_writer.as_default():
        # write summary of epoch loss
        tf.summary.scalar("classifier/test_accuracy", metric_test_accuracy_epochs.result(), step=0)
        tf.summary.scalar("classifier/test_f1", metric_test_f1_epochs.result(), step=0)

    global_labels = tf.concat(global_labels, axis=0)
    global_predictions = tf.concat(global_predictions, axis=0)
    class_report = classification_report(global_labels, global_predictions, target_names=dataset.LABELS)
    logger.info("Classification Report on TEST set:\n{}".format(class_report))


def embed_triplet(anchor, neighbour, opposite):
    if model_embedding is not None:
        emb_anchor = model_embedding(anchor, training=False)
        emb_neighbour = model_embedding(neighbour, training=False)
        emb_opposite = model_embedding(opposite, training=False)
    else:
        emb_anchor = anchor
        emb_neighbour = neighbour
        emb_opposite = opposite

    return emb_anchor, emb_neighbour, emb_opposite


def get_predictions_from_triplet(batch):
    # unpack batch
    anchor, neighbour, opposite, triplet_metadata = batch

    # embed the triplets into the embedding space
    emb_anchor, emb_neighbour, emb_opposite = embed_triplet(anchor, neighbour, opposite)

    # retrieve labels from metadata
    triplet_labels = tf.strings.to_number(triplet_metadata[:, 0], tf.float32)

    # run the forward pass of the layer
    pred_anchor = classifier(emb_anchor)
    pred_neighbour = classifier(emb_neighbour)
    pred_opposite = classifier(emb_opposite)

    pred = tf.concat([pred_anchor, pred_neighbour, pred_opposite], axis=0)
    labels = tf.concat([triplet_labels[:, 0], triplet_labels[:, 1], triplet_labels[:, 2]], axis=0)
    labels = tf.dtypes.cast(labels, tf.int32)

    return pred, labels


if __name__ == "__main__":
    set_gpu_experimental_growth()

    # load the arguments
    args = parser.parse_args()

    # load default config file for classifier
    json_classifier = os.path.join(args.experiment_dir, "config", "params.json")
    params_classifier = Params(json_classifier)

    if args.model_to_load != "None":
        # load the parameters of the model to train
        # set the existing model to the experiment path
        experiment_path = os.path.join(args.experiment_dir, args.dataset_dir, args.model_to_load)
        # load folder for saving model
        saved_model_path = os.path.join(experiment_path, "saved_model")

        # get the parameters from the json file
        json_saved_model = os.path.join(experiment_path, "logs", "params.json")
        params_saved_model = Params(json_saved_model)

        # load the embedding model
        model_embedding = ModelFactory.create_model(params_saved_model.model, params=params_saved_model)
        # define checkpoint and checkpoint manager
        ckpt = tf.train.Checkpoint(net=model_embedding)
        manager = tf.train.CheckpointManager(ckpt, saved_model_path, max_to_keep=3)

        # check if models_embedding has been trained before
        ckpt.restore(manager.latest_checkpoint)
        if not manager.latest_checkpoint:
            raise ValueError("Embedding model could not be restored")
    else:
        # load default parameters
        params_saved_model = params_classifier
        # no embedding model should be used, e.g. as a baseline model
        model_embedding = None

    # define dataset
    dataset = DatasetFactory.create_dataset(name=params_saved_model.dataset, params=params_saved_model)
    # get the feature extractor from the factory
    extractor = ExtractorFactory.create_extractor(params_saved_model.feature_extractor, params=params_saved_model)
    # define triplet input pipeline
    pipeline = TripletsInputPipeline(params=params_saved_model, dataset=dataset)

    # create the classifier model
    if args.classifier_type == "Dense":
        classifier = ClassifierDense("DenseClassifier", n_labels=len(dataset.LABELS))
    elif args.classifier_type == "Logistic":
        classifier = ClassifierLogistic("LogisticClassifier", n_labels=len(dataset.LABELS))
    else:
        raise ValueError("Wrong specified classifier")

    # loss for the classifier
    classifier_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # create the optimizer for the classifier
    classifier_optimizer = tf.keras.optimizers.Adam(learning_rate=params_classifier.learning_rate)

    # create folders for experiment results
    experiment_name = "{0}-{1}".format(classifier.model_name, params_classifier.experiment_name)
    experiment_path, log_path, tensorb_path, save_path = Utils.create_load_folders_for_experiment(args,
                                                                                                  dataset_folder=dataset.EXPERIMENT_FOLDER,
                                                                                                  model_name=experiment_name)

    # set logger
    Utils.set_logger(log_path, params_classifier.log_level)
    logger = logging.getLogger("Main ({})".format(params_classifier.experiment_name))

    # print params
    logger.info("PARAMS CLASSIFIER:")
    params_classifier.print(json_classifier, logger=logger)
    logger.info("Model loaded: {}".format(args.model_to_load))

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

    metric_test_accuracy_epochs = tf.keras.metrics.SparseCategoricalAccuracy()
    metric_test_f1_epochs = tfa.metrics.F1Score(num_classes=len(dataset.LABELS), average="macro")

    # define checkpoint and checkpoint manager
    ckpt_classifier = tf.train.Checkpoint(optimizer=classifier_optimizer, step=tf.Variable(1), net=classifier)
    manager_classifier = tf.train.CheckpointManager(ckpt_classifier, save_path, max_to_keep=3)

    for epoch in range(params_classifier.epochs):
        logger.info("Starting epoch {0} from {1}".format(epoch + 1, params_classifier.epochs))

        # train the classifier
        training()
        # evaluate the classifier
        evaluating()

        # reset metrics every epoch
        metric_train_accuracy_batches.reset_states()
        metric_train_accuracy_epochs.reset_states()
        metric_eval_accuracy_epochs.reset_states()

        metric_train_loss_batches.reset_states()
        metric_train_loss_epochs.reset_states()
        metric_eval_loss_epochs.reset_states()

        metric_train_f1_batches.reset_states()
        metric_train_f1_epochs.reset_states()
        metric_eval_f1_epochs.reset_states()

        # save model after each epoch
        manager_save_path = manager_classifier.save()
        logger.info("Saved checkpoint for epoch {0}: {1}".format(epoch + 1, manager_save_path))

        # reinitialise pipeline after epoch
        pipeline.reinitialise()
        logger.info("Epoch end")

    if args.eval_on_test_at_end == "True":
        testing()
