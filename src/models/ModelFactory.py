import logging
from typing import Callable

from src.models.BaseModel import BaseModel


class ModelFactory:
    """ The factory class for creating various models. """

    # internal registry for available models
    registry = {}
    # logger for status information
    logger = logging.getLogger(__name__)

    @classmethod
    def register(cls, name: str) -> Callable:
        """ Class method to register Model classes to the internal registry.
        Args:
            name (str): The name of the executor.
        Returns:
            The Executor class itself.
        """

        def inner_wrapper(wrapped_class: BaseModel) -> Callable:
            if name in cls.registry:
                cls.logger.warning('Model {} already exists, will replace it'.format(name))
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create_model(cls, name: str, **kwargs) -> BaseModel:
        """ Factory command to create the model.
        This method gets the appropriate Model class from the registry
        and creates an instance of it, while passing in the parameters
        given in ``kwargs``.
        Args:
            name (str): The name of the model to create.
        Returns:
            An instance of the model that is created.
        """

        if name not in cls.registry:
            raise ValueError("Model {} does not exist in the registry".format(name))

        exec_class = cls.registry[name]
        model = exec_class(**kwargs)
        return model
