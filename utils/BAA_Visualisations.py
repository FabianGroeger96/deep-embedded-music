# %%
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

# %%
y, sr = librosa.load(librosa.util.example_audio_file(), offset=30, duration=10)
librosa.feature.mfcc(y=y, sr=sr)

# %%
# Get more components
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# %%
# Visualize the MFCC series
plt.figure(figsize=(15, 5))
librosa.display.specshow(mfccs, x_axis='time')

clb = plt.colorbar()
clb.set_label('value of MFCC coefficient', fontsize=18)

plt.yticks(np.arange(0, 14, 1.0))
plt.xlabel('Time (s)', fontsize=18)
plt.ylabel('MFCC coefficients', fontsize=18)
plt.title('MFCCs', fontsize=22)
plt.show()

# %%
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
plt.figure(figsize=(15, 5))
librosa.display.specshow(D, y_axis='linear', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram', fontsize=22)
plt.xlabel('Time (s)', fontsize=18)
plt.ylabel('Frequency (Hz)', fontsize=18)
plt.show()

# %%
plt.figure(figsize=(15, 5))
librosa.display.waveplot(y, sr=sr)
plt.title('Waveplot', fontsize=22)
plt.xlabel('Time (s)', fontsize=18)
plt.ylabel('Amplitude', fontsize=18)
plt.show()
