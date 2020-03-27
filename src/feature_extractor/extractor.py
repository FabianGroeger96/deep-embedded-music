from abc import ABC, abstractmethod

import tensorflow as tf

from src.utils.params import Params


class Extractor(ABC):
    def __init__(self, params: Params):
        self.sample_rate = params.sample_rate
        self.sample_size = params.sample_size

        self.lower_edge_hertz = 50
        self.upper_edge_hertz = self.get_nyquist_frequency()

    @abstractmethod
    def extract(self, audio):
        pass

    @abstractmethod
    def get_output_shape(self):
        pass

    def get_nyquist_frequency(self):
        return self.sample_rate / 2

    # compute STFT
    # INPUT : (sample_size, )
    # OUTPUT: (frame_size, fft_size // 2 + 1)
    def get_stft_spectrogram(self, data, frame_length, frame_step, fft_size):
        # Input: A Tensor of [batch_size, num_samples]
        # mono PCM samples in the range [-1, 1].
        stfts = tf.signal.stft(data,
                               frame_length=frame_length,
                               frame_step=frame_step,
                               fft_length=fft_size)

        # determine the amplitude
        spectrograms = tf.abs(stfts)

        return spectrograms

    # compute mel-Frequency
    # INPUT : (frame_size, fft_size // 2 + 1)
    # OUTPUT: (frame_size, mel_bin_size)
    def get_mel(self, stfts, n_mel_bin):
        # STFT-bin
        # 257 (= FFT size / 2 + 1)
        n_stft_bin = stfts.shape[-1]

        # linear_to_mel_weight_matrix shape: (257, 128)
        # (FFT size / 2 + 1, num of mel bins)
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=n_mel_bin,
            num_spectrogram_bins=n_stft_bin,
            sample_rate=self.sample_rate,
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
    def get_mfcc(self, log_mel_spectrograms, n_mfcc_bin):
        mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)
        mfcc = mfcc[..., :n_mfcc_bin]

        return mfcc

    # OUTPUT: (frame_size, mel_bin_size)
    def extract_log_mel_features(self, audio, frame_length, frame_step, fft_size, n_mel_bin):
        # find the feature value (STFT)
        stft = self.get_stft_spectrogram(audio,
                                         frame_length=frame_length,
                                         frame_step=frame_step,
                                         fft_size=fft_size)
        # find features (logarithmic mel spectrogram)
        log_mel_spectrogram = self.get_mel(stft, n_mel_bin)

        return log_mel_spectrogram

    # OUTPUT: (frame_size, n_mfcc_bin)
    def extract_mfcc_features(self, audio, frame_length, frame_step, fft_size, n_mel_bin, n_mfcc_bin):
        # extract the log mel features from the audio
        log_mel_spectrogram = self.extract_log_mel_features(audio,
                                                            frame_length=frame_length,
                                                            frame_step=frame_step,
                                                            fft_size=fft_size,
                                                            n_mel_bin=n_mel_bin)
        # extract MFCC feature values
        mfcc = self.get_mfcc(log_mel_spectrogram, n_mfcc_bin)

        return mfcc
