import librosa
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from librosa import display

from hparams import hparams
from utils.audio import load_wav, plot_spectrogram, linear_scale_spectrogram, ms_to_samples, \
    mel_scale_spectrogram, calculate_mfccs, plot_feature_frames

wav_path = '/home/yves-noel/documents/master/projects/datasets/timit/TIMIT/TRAIN/DR1/FCJF0/SA1.WAV'
n_fft = hparams.n_fft

wav, sr = load_wav(wav_path, sampling_rate=None)

print("len(wav) = {}s".format(len(wav) / sr))

win_len = ms_to_samples(hparams.win_len, sampling_rate=sr)
hop_len = ms_to_samples(hparams.win_hop, sampling_rate=sr)

linear_spec = linear_scale_spectrogram(wav,
                                       n_fft,
                                       hop_length=hop_len,
                                       win_length=win_len)

mel_spec = mel_scale_spectrogram(wav, n_fft, sampling_rate=sr,
                                 hop_length=hop_len,
                                 win_length=win_len,
                                 n_mels=hparams.n_mels,
                                 fmin=hparams.mel_fmin,
                                 fmax=hparams.mel_fmax,
                                 power=2)

# Convert the linear-scaled power spectrogram to its decibel representation.
linear_spec = linear_spec ** 2
linear_spec_db = librosa.power_to_db(linear_spec, ref=np.max)

# Convert the Mel-scaled power spectrogram to its decibel representation.
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# Calculate mfcc features.
mfccs = calculate_mfccs(mel_spec_db, sr, hparams.n_mfcc)

# plot_waveform(wav, sampling_rate=sr)
#
# plot_spectrogram(linear_spec_db,
#                  sampling_rate=sr,
#                  hop_length=hop_len,
#                  fmin=hparams.mel_fmin,
#                  fmax=hparams.mel_fmax,
#                  y_axis='linear',
#                  title='Linear-scale power spectrogram')
#
# plot_spectrogram(mel_spec_db,
#                  sampling_rate=sr,
#                  hop_length=hop_len,
#                  fmin=hparams.mel_fmin,
#                  fmax=hparams.mel_fmax,
#                  y_axis='mel',
#                  title='Mel-scale power spectrogram')

# Calculate mfcc delta features.
mfccs_d1 = librosa.feature.delta(mfccs, width=3, order=1)
# Calculate mfcc delta-delta features.
mfccs_d2 = librosa.feature.delta(mfccs, width=3, order=2)

features = np.vstack((mfccs, mfccs_d1, mfccs_d2))

plot_feature_frames(mfccs, sampling_rate=sr, hop_length=hop_len, title='MFCC')
plot_feature_frames(mfccs_d1, sampling_rate=sr, hop_length=hop_len, title=r'MFCC-$\Delta$')
plot_feature_frames(mfccs_d2, sampling_rate=sr, hop_length=hop_len, title=r'MFCC-$\Delta^2$')
