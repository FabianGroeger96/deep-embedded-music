import numpy as np
import tensorflow as tf

from src.feature_extractor.base_extractor import BaseExtractor
from src.feature_extractor.extractor_factory import ExtractorFactory
from src.utils.params import Params


@ExtractorFactory.register("MFCCExtractor")
class MFCCExtractor(BaseExtractor):
    """ The extractor for representing an audio in the form of the MFCCs. """

    def __init__(self, params: Params):
        """
        Initialises the extractor, by passing in the global hyperparameters.

        :param params: the global hyperparameters for initialising the dataset.
        """
        super().__init__(params=params)

    def extract(self, audio):
        """
        Extracts the MFCCs from the given audio.
        Audio can be mono or contain multiple channels.

        OUTPUT: (frame_size, n_mfcc_bin, ?channels)

        :return: the extracted feature from the given audio.
        """
        if len(audio.shape) == 1:
            log_mel_spectrogram = self.extract_mfcc_features(audio)

            return log_mel_spectrogram

        elif len(audio.shape) == 2:
            # list of features
            feature_list = []
            # loop over the number of channels within the audio signal
            for i in range(audio.shape[-1]):
                # extract one audio channel
                audio_channel = tf.squeeze(audio[:, i])
                # find the feature value (STFT)
                log_mel_spectrogram = self.extract_mfcc_features(audio_channel)
                # append the feature to the list of features
                feature_list.append(log_mel_spectrogram)
            # stack the different audio channels on top of each other (..., ..., channels)
            audio_features_extracted = tf.stack(feature_list, axis=-1)

            return audio_features_extracted

        else:
            raise ValueError("Audio has wrong shape.")

    def get_output_shape(self):
        """
        Returns the shape of the MFCCs, by using the provided hyperparameters.

        :return: the output shape of the extracted feature.
        """
        # calculate the frame size
        data = np.zeros(self.params.sample_rate * self.params.sample_tile_size)
        # shape: (frame_size, frame_length)
        frames = tf.signal.frame(data, frame_length=self.params.frame_length, frame_step=self.params.frame_step)
        frame_size = frames.shape[0]

        return frame_size, self.params.n_mfcc_bin
