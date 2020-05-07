from abc import ABC, abstractmethod

import tensorflow as tf

from src.utils.params import Params


class BaseExtractor(ABC):
    def __init__(self, params: Params):
        self.params = params

        self.lower_edge_hertz = 50
        self.upper_edge_hertz = self.get_nyquist_frequency()

    @abstractmethod
    def extract(self, audio):
        pass

    @abstractmethod
    def get_output_shape(self):
        pass

    def get_nyquist_frequency(self):
        return self.params.sample_rate / 2

    # compute STFT
    # INPUT : (sample_size, )
    # OUTPUT: (frame_size, fft_size // 2 + 1)
    def get_stft_spectrogram(self, data):
        # Input: A Tensor of [batch_size, num_samples]
        # mono PCM samples in the range [-1, 1].
        stfts = tf.signal.stft(data,
                               frame_length=self.params.frame_length,
                               frame_step=self.params.frame_step,
                               fft_length=self.params.fft_size)

        # determine the amplitude
        spectrograms = tf.abs(stfts)

        return spectrograms

    # compute mel-Frequency
    # INPUT : (frame_size, fft_size // 2 + 1)
    # OUTPUT: (frame_size, mel_bin_size)
    def get_mel(self, stfts):
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

    # compute MFCC
    # INPUT : (frame_size, mel_bin_size)
    # OUTPUT: (frame_size, n_mfcc_bin)
    def get_mfcc(self, log_mel_spectrograms):
        mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)
        mfcc = mfcc[..., :self.params.n_mfcc_bin]

        return mfcc

    # OUTPUT: (frame_size, mel_bin_size)
    def extract_log_mel_features(self, audio):
        # find the feature value (STFT)
        stft = self.get_stft_spectrogram(audio)
        # find features (logarithmic mel spectrogram)
        log_mel_spectrogram = self.get_mel(stft)

        return log_mel_spectrogram

    # OUTPUT: (frame_size, n_mfcc_bin)
    def extract_mfcc_features(self, audio):
        # extract the log mel features from the audio
        log_mel_spectrogram = self.extract_log_mel_features(audio)
        # extract MFCC feature values
        mfcc = self.get_mfcc(log_mel_spectrogram)

        return mfcc
