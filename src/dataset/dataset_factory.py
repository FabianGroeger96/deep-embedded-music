import logging
from typing import Callable

from src.dataset.base_dataset import BaseDataset


class DatasetFactory:
    """ The factory class for creating various dataset. """

    # internal registry for available dataset
    registry = {}
    # logger for status information
    logger = logging.getLogger(__name__)

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Class method to register dataset classes to the internal registry.
        :param name: the name of the dataset.
        :return: the dataset itself.
        """

        def inner_wrapper(wrapped_class: BaseDataset) -> Callable:
            if name in cls.registry:
                cls.logger.warning("Dataset {} already exists, will replace it".format(name))
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create_dataset(cls, name: str, **kwargs) -> BaseDataset:
        """
        Factory command to create the dataset.
        This method gets the appropriate dataset class from the registry
        and creates an instance of it, while passing in the parameters
        given in ``kwargs``.
        :param name: the name of the dataset to create.
        :param kwargs: the parameters to instantiate the dataset.
        :return: an instance of the dataset which is created.
        """

        if name not in cls.registry:
            raise ValueError("Dataset {} does not exist in the registry".format(name))

        exec_class = cls.registry[name]
        model = exec_class(**kwargs)
        return model
