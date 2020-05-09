from abc import ABC, abstractmethod

import tensorflow as tf

from src.utils.params import Params


class BaseExtractor(ABC):
    """
    The abstract base class of the extractors.
    The extractors are responsible to represent audio in different ways.
    """

    def __init__(self, params: Params):
        """
        Initialises the extractor, by passing in the global hyperparameters.

        :param params: the global hyperparameters for initialising the dataset.
        """
        self.params = params

        self.lower_edge_hertz = 50
        self.upper_edge_hertz = self.get_nyquist_frequency()

    @abstractmethod
    def extract(self, audio):
        """ Extracts the implemented feature representation from an audio. """
        pass

    @abstractmethod
    def get_output_shape(self):
        """ Returns the shape of the feature representation, which will be implemented. """
        pass

    def get_nyquist_frequency(self):
        """ Calculates and returns the nyquist frequency, which indicates what the maximal perceived frequency is. """
        return self.params.sample_rate / 2

    def get_stft_spectrogram(self, data):
        """
        Calculates the STFT of the given audio data tensor, using the global hyperparameters.

        INPUT : Tensor of (sample_size, )
        OUTPUT: (frame_size, fft_size // 2 + 1)

        :return: the calculated spectrogram of the STFT.
        """
        stfts = tf.signal.stft(data,
                               frame_length=self.params.frame_length,
                               frame_step=self.params.frame_step,
                               fft_length=self.params.fft_size)

        # determine the amplitude
        spectrograms = tf.abs(stfts)

        return spectrograms

    def get_mel(self, stfts):
        """
        Calculates the log Mel frequency spectrogram of the given STFT spectrogram.

        INPUT : (frame_size, fft_size // 2 + 1)
        OUTPUT: (frame_size, mel_bin_size)

        :return: the calculated log Mel frequency spectrogram.
        """
        # STFT-bin
        # 257 (= FFT size / 2 + 1)
        n_stft_bin = stfts.shape[-1]

        # linear_to_mel_weight_matrix shape: (257, 128)
        # (FFT size / 2 + 1, num of mel bins)
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.params.n_mel_bin,
            num_spectrogram_bins=n_stft_bin,
            sample_rate=self.params.sample_rate,
            lower_edge_hertz=self.lower_edge_hertz,
            upper_edge_hertz=self.upper_edge_hertz
        )

        # mel_spectrograms shape: (1, 98, 128)
        mel_spectrograms = tf.tensordot(
            stfts,  # (1, 98, 257)
            linear_to_mel_weight_matrix,  # (257, 128)
            1)

        # take the logarithm (add a small number to avoid log(0))
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

        return log_mel_spectrograms

    def get_mfcc(self, log_mel_spectrograms):
        """
        Calculates the MFCCs of the given log Mel frequency spectrogram.

        INPUT : (frame_size, mel_bin_size)
        OUTPUT: (frame_size, n_mfcc_bin)

        :return: the calculated MFCCs.
        """
        mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)
        mfcc = mfcc[..., :self.params.n_mfcc_bin]

        return mfcc

    def extract_log_mel_features(self, audio):
        """
        Extracts the log Mel frequency spectrogram of the given audio tensor.
        1. calculating the STFT
        2. extracting log Mel frequency spectrogram

        OUTPUT: (frame_size, mel_bin_size)

        :return: the calculated log Mel frequency spectrogram.
        """
        # find the feature value (STFT)
        stft = self.get_stft_spectrogram(audio)
        # find features (logarithmic mel spectrogram)
        log_mel_spectrogram = self.get_mel(stft)

        return log_mel_spectrogram

    def extract_mfcc_features(self, audio):
        """
        Extracts the MFCCs of a given audio tensor.
        1. calculating the STFT
        2. extracting log Mel frequency spectrogram
        3. extracting MFCC

        OUTPUT: (frame_size, n_mfcc_bin)

        :return: the calculated MFCCs.
        """
        # extract the log mel features from the audio
        log_mel_spectrogram = self.extract_log_mel_features(audio)
        # extract MFCC feature values
        mfcc = self.get_mfcc(log_mel_spectrogram)

        return mfcc
