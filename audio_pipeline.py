import argparse
import os

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

    logger = set_logger(experiment_path)

    pipeline = audio_pipeline = TripletsInputPipeline(
        audio_files_path=params.audio_files_path,
        info_file_path=params.info_file_path,
        sample_rate=params.sample_rate,
        sample_size=params.sample_size,
        batch_size=params.batch_size,
        prefetch_batches=params.prefetch_batches,
        input_processing_buffer_size=params.input_processing_buffer_size)
    pipeline.generate_samples()
