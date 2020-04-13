import json
import os


class Params:
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.experiment_name = None

        self.dcase_dataset_path = None
        self.dcase_dataset_fold = None

        self.music_dataset_path = None

        self.log_level = None

        self.model = None
        self.dataset = None

        self.save_model = None
        self.saved_model_path = None
        self.save_frequency = None

        self.use_profiler = None

        self.epochs = None
        self.batch_size = None
        self.prefetch_batches = None
        self.random_selection_buffer_size = None
        self.learning_rate = None

        self.shuffle_dataset = None
        self.train_test_split = None

        self.gen_count = None

        self.sample_rate = None
        self.sample_size = None
        self.sample_tile_size = None
        self.sample_tile_neighbourhood = None

        self.stereo_channels = None  # only for DCASE dataset
        self.to_mono = None  # leave at TRUE for MusicDataset

        self.feature_extractor = None
        self.frame_length = None
        self.frame_step = None
        self.fft_size = None
        self.n_mel_bin = None
        self.n_mfcc_bin = None

        self.margin = None
        self.embedding_size = None

        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)

        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def print(self, json_path, logger):
        with open(json_path) as f:
            params = json.load(f)
            logger.info(json.dumps(params, indent=4, sort_keys=True))

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__
