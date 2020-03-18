import numpy as np
import tensorflow as tf

from feature_extractor.extractor import Extractor


class LogMelExtractor(Extractor):
    def __init__(self, sample_rate, sample_size, frame_length, frame_step, fft_size, n_mel_bin):
        super().__init__(sample_rate=sample_rate, sample_size=sample_size)

        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_size = fft_size
        self.n_mel_bin = n_mel_bin

    # OUTPUT: (frame_size, mel_bin_size, channels)
    def extract(self, audio):
        # extract the number of channels within the audio signal
        channels = audio.shape[-1]

        # list of features
        feature_list = []

        for i in range(channels):
            # extract one audio channel
            audio_channel = audio[:, i]
            audio_channel = tf.squeeze(audio_channel)

            # find the feature value (STFT)
            stft = self.get_stft_spectrogram(audio_channel,
                                             frame_length=self.frame_length,
                                             frame_step=self.frame_step,
                                             fft_size=self.fft_size)

            # find features (logarithmic mel spectrogram)
            log_mel_spectrogram = self.get_mel(stft, self.n_mel_bin)

            # append the feature to the list of features
            feature_list.append(log_mel_spectrogram)

        audio_features_extracted = tf.stack(feature_list, axis=-1)

        return audio_features_extracted

    def get_output_shape(self):
        # calculate the frame size
        data = np.zeros(self.sample_rate)
        # shape: (frame_size, frame_length)
        frames = tf.signal.frame(data, frame_length=self.frame_length, frame_step=self.frame_step)
        frame_size = frames.shape[0]

        return frame_size, self.n_mel_bin
