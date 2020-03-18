import argparse
import os

from feature_extractor.log_mel_extractor import LogMelExtractor
from input_pipeline.triplet_input_pipeline import TripletsInputPipeline
from utils.params import Params
from utils.utils import set_logger

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/triplet_loss',
                    help="Experiment directory containing params.json")

if __name__ == "__main__":
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    experiment_path = os.path.join(args.model_dir, params.experiment_name)
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    logger = set_logger(experiment_path, params.log_level)

    pipeline = audio_pipeline = TripletsInputPipeline(
        dataset_path=params.audio_files_path,
        fold=params.fold,
        sample_rate=params.sample_rate,
        sample_size=params.sample_size,
        batch_size=params.batch_size,
        prefetch_batches=params.prefetch_batches,
        random_selection_buffer_size=params.random_selection_buffer_size)

    feature_extractor = LogMelExtractor(sample_rate=16000, fft_size=512, n_mel_bin=128)

    for anchor, neighbour, opposite, triplet_labels in audio_pipeline.get_dataset(feature_extractor,
                                                                                  shuffle=bool(params.shuffle_dataset),
                                                                                  calc_dist=bool(params.calc_dist)):
        print(anchor)
        print(neighbour)
        print(opposite)
        print(triplet_labels)
