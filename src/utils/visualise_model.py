import os

import tensorflow as tf
from tensorboard.plugins import projector

from src.input_pipeline.dcase_data_frame import DCASEDataFrame


def save_labels_tsv(labels, filepath, log_dir):
    with open(os.path.join(log_dir, filepath), 'w') as f:
        for label in labels.numpy():
            f.write('{}\n'.format(DCASEDataFrame.LABELS[int(label)]))


def save_embeddings_tsv(embeddings, filepath, log_dir):
    with open(os.path.join(log_dir, filepath), 'w') as f:
        for embedding in embeddings.numpy():
            for vec in embedding:
                f.write('{}\t'.format(vec))
            f.write('\n')


def visualise_embeddings(embeddings, triplet_labels, tensorboard_path, step):
    # reshape the embedding to a flat tensor (batch_size, ?)
    # emb = tf.reshape(embeddings, [embeddings.shape[0], -1])
    tensor_embeddings = tf.Variable(embeddings, name='embeddings')

    # save labels and embeddings to .tsv, to assign each embedding a label
    save_labels_tsv(triplet_labels[0], 'labels.tsv', tensorboard_path)
    save_embeddings_tsv(embeddings, 'embeddings.tsv', tensorboard_path)

    # save the embeddings to a checkpoint file, which will then be loaded by the projector
    saver = tf.compat.v1.train.Saver([tensor_embeddings])
    saver.save(sess=None, global_step=step, save_path=os.path.join(tensorboard_path, "embeddings.ckpt"))

    # register projector
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "embeddings"
    embedding.metadata_path = "labels.tsv"
    projector.visualize_embeddings(tensorboard_path, config)