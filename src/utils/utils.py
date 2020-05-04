import logging
import os
import pathlib
import shutil
from typing import Union
from datetime import datetime


class Utils:
    """ Contains various util functions for the project. """

    @staticmethod
    def get_files_in_path(path: Union[str, pathlib.Path], file_extension: str):
        """
        Gets all the files within a given path of a given file extension.

        :param path: the path to search the files.
        :param file_extension: all files with this file extension will be returned.
        :return: the files within the path.
        """
        files = []
        for root_dir, dir_names, file_names in os.walk(path):
            # ignore tmp files
            file_names = [f for f in file_names if not f.startswith(".")]
            # find only files with given extension
            file_names = [f for f in file_names if f.endswith(file_extension)]

            for file in file_names:
                files.append(os.path.join(root_dir, file))

        return files

    @staticmethod
    def check_if_path_exists(path: Union[str, pathlib.Path]):
        """
        Checks if a given path exists.

        :param path: the path to check.
        :return: the path which was checked.
        :raises: ValueError: if the path is not valid, does not exist.
        """
        path = pathlib.Path(path)
        if not path.exists():
            raise ValueError(
                "Path {} is not valid. Please provide an existing path".format(path))

        return path

    @staticmethod
    def set_logger(log_path, log_level: str = "INFO"):
        """
        Sets the logger to log info in terminal and file `log_path`.

        In general, it is useful to have a logger so that every output to the terminal is saved
        in a permanent file. Here we save it to `log_path/experiment.log`.

        :param log_path: (string) where to log
        :param log_level: sets the log level
        """
        logger = logging.getLogger()
        logger.setLevel(log_level)

        if not logger.handlers:
            # Logging to a file
            file_handler = logging.FileHandler(os.path.join(log_path, "experiment.log"))
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)-22s - %(levelname)-8s - %(message)s"))
            logger.addHandler(file_handler)

            # Logging to console
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)-22s - %(levelname)-8s - %(message)s"))
            logger.addHandler(stream_handler)

        return logger

    @staticmethod
    def create_load_folders_for_experiment(args, dataset_folder, model_name, saved_model_path="", copy_json_file=True):
        """
        Creates or loads the folders for a experiment.

        Will create or load the experiment, logs, tensorboard and saved model folder.

        If a saved_model_path is provided, the method will load the folders.
        If not it will create the respective folders.

        :param args: the arguments of the current experiment.
        :param dataset_folder: the folder of the experiments for the specified dataset.
        :param model_name: the name of the current model used in the experiment.
        :param saved_model_path: the saved model path, if the folders should be loaded.
        :param copy_json_file: if the global experiment json file should be copied.
        :return: the experiment, logs, tensorboard and saved model folder path.
        """
        if saved_model_path is "":
            # create experiment name folder
            Utils.create_folder(os.path.join(args.experiment_dir, dataset_folder))
            experiment_path = Utils.create_folder(os.path.join(args.experiment_dir, dataset_folder, "results"))
            # create current experiment folder
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            experiment_folder_name = "{0}-{1}".format(model_name, current_time)
            experiment_path = Utils.create_folder(os.path.join(experiment_path, experiment_folder_name))
        else:
            # set the existing model to the experiment path
            experiment_path = os.path.join(args.experiment_dir, dataset_folder, saved_model_path)

        # create folder for logging
        logs_path = Utils.create_folder(os.path.join(experiment_path, "logs"))
        # create folder for tensorboard
        tensorboard_path = Utils.create_folder(os.path.join(experiment_path, "tensorboard"))
        # create folder for saving model
        saved_model_path = Utils.create_folder(os.path.join(experiment_path, "saved_model"))
        # copy current param file to log directory
        if copy_json_file:
            params_path = os.path.join(args.experiment_dir, "config", "params.json")
            shutil.copyfile(params_path, os.path.join(logs_path, "params.json"))

        return experiment_path, logs_path, tensorboard_path, saved_model_path

    @staticmethod
    def create_folder(path):
        """
        Creates a folder at a given path.
        :param path: the folder path to create.
        :return: the path of the newly created path.
        """
        if not os.path.exists(path):
            os.mkdir(path)

        return path
