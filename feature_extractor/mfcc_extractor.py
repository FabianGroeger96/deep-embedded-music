import tensorflow as tf

from feature_extractor.extractor import Extractor


class MFCCExtractor(Extractor):
    def __init__(self, sample_rate, fft_size, n_mel_bin, n_mfcc_bin):
        super().__init__(sample_rate=sample_rate)

        self.fft_size = fft_size
        self.n_mel_bin = n_mel_bin
        self.n_mfcc_bin = n_mfcc_bin

    # OUTPUT: (frame_size, n_mfcc_bin)
    def extract(self, audio):
        # find the feature value (STFT)
        stfts = self.get_stft_spectrogram(audio, self.fft_size)

        # find features (logarithmic mel spectrogram)
        log_mel_spectrograms = self.get_mel(stfts, self.n_mel_bin)

        # finding feature values (MFCC)
        mfcc = self.get_mfcc(log_mel_spectrograms, self.n_mfcc_bin)

        return mfcc

    # compute STFT
    # INPUT : (sample_size, )
    # OUTPUT: (frame_size, fft_size // 2 + 1)
    def get_stft_spectrogram(self, data, fft_size):
        # Input: A Tensor of [batch_size, num_samples]
        # mono PCM samples in the range [-1, 1].
        stfts = tf.signal.stft(data,
                               frame_length=480,
                               frame_step=160,
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
            lower_edge_hertz=0.0,
            upper_edge_hertz=8000.0
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
