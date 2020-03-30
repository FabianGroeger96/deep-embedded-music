import logging
from typing import Callable

from src.feature_extractor.base_extractor import BaseExtractor


class ExtractorFactory:
    """ The factory class for creating various extractors. """

    # internal registry for available extractors
    registry = {}
    # logger for status information
    logger = logging.getLogger(__name__)

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Class method to register extractor classes to the internal registry.
        :param name: the name of the extractor.
        :return: the extractor itself.
        """

        def inner_wrapper(wrapped_class: BaseExtractor) -> Callable:
            if name in cls.registry:
                cls.logger.warning("Extractor {} already exists, will replace it".format(name))
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create_extractor(cls, name: str, **kwargs) -> BaseExtractor:
        """
        Factory command to create the extractor.
        This method gets the appropriate extractor class from the registry
        and creates an instance of it, while passing in the parameters
        given in ``kwargs``.
        :param name: the name of the extractor to create.
        :param kwargs: the parameters to instantiate the extractor.
        :return: an instance of the extractor which is created.
        """

        if name not in cls.registry:
            raise ValueError("Extractor {} does not exist in the registry".format(name))

        exec_class = cls.registry[name]
        extractor = exec_class(**kwargs)
        return extractor
