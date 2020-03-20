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

    # OUTPUT: (frame_size, mel_bin_size, ?channels)
    def extract(self, audio):
        if len(audio.shape) == 1:
            log_mel_spectrogram = self.extract_log_mel_features(audio,
                                                                frame_length=self.frame_length,
                                                                frame_step=self.frame_step,
                                                                fft_size=self.fft_size,
                                                                n_mel_bin=self.n_mel_bin)

            return log_mel_spectrogram

        elif len(audio.shape) == 2:
            # list of features
            feature_list = []
            # loop over the number of channels within the audio signal
            for i in range(audio.shape[-1]):
                # extract one audio channel
                audio_channel = tf.squeeze(audio[:, i])
                # find the feature value (STFT)
                log_mel_spectrogram = self.extract_log_mel_features(audio_channel,
                                                                    frame_length=self.frame_length,
                                                                    frame_step=self.frame_step,
                                                                    fft_size=self.fft_size,
                                                                    n_mel_bin=self.n_mel_bin)
                # append the feature to the list of features
                feature_list.append(log_mel_spectrogram)
            # stack the different audio channels on top of each other (..., ..., channels)
            audio_features_extracted = tf.stack(feature_list, axis=-1)

            return audio_features_extracted

        else:
            raise ValueError("Audio has wrong shape.")

    def get_output_shape(self):
        # calculate the frame size
        data = np.zeros(self.sample_rate)
        # shape: (frame_size, frame_length)
        frames = tf.signal.frame(data, frame_length=self.frame_length, frame_step=self.frame_step)
        frame_size = frames.shape[0]

        return frame_size, self.n_mel_bin
