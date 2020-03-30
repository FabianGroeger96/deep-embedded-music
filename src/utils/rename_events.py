import os
from pathlib import Path

import tensorflow as tf
from tensorflow.core.util.event_pb2 import Event

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Not necessary to use GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Avoid log messages


def rename_events(input_path, output_path, old_tags, new_tag):
    # Make a record writer
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        # Iterate event records
        for rec in tf.data.TFRecordDataset([str(input_path)]):
            # Read event
            ev = Event()
            ev.MergeFromString(rec.numpy())
            # Check if it is a summary
            if ev.summary:
                # Iterate summary values
                for v in ev.summary.value:
                    # Check if the tag should be renamed
                    if v.tag in old_tags:
                        # Rename with new tag name
                        v.tag = new_tag
            writer.write(ev.SerializeToString())


def rename_events_dir(input_dir, output_dir, old_tags, new_tag):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    # Make output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    # Iterate event files
    for ev_file in input_dir.glob('**/*.tfevents*'):
        # output event file
        out_file = Path(output_dir, ev_file.relative_to(input_dir))
        # Write renamed events
        rename_events(ev_file, out_file, old_tags, new_tag)


if __name__ == '__main__':
    input_dir = "experiments/DCASE/results/ConvNet1D-20200326-065709/tensorboardV2"
    output_dir = "experiments/DCASE/results/ConvNet1D-20200326-065709/tensorboardV3"
    old_tags = "triplet_loss/epochs"
    new_tag = "triplet_loss/loss_triplet_epochs"

    old_tags = old_tags.split(';')
    rename_events_dir(input_dir, output_dir, old_tags, new_tag)
    print('Done')
