import librosa
import numpy as np

from hparams import hparams
from utils.audio import load_wav, plot_spectrogram, linear_scale_spectrogram, ms_to_samples, \
    mel_scale_spectrogram

wav_path = '/home/yves-noel/documents/master/projects/datasets/timit/TIMIT/TRAIN/DR1/FCJF0/SA1.WAV'
n_fft = hparams.n_fft

wav, sr = load_wav(wav_path, sampling_rate=None)

print("len(wav) = {}s".format(len(wav) / sr))

win_len = ms_to_samples(hparams.win_len, sampling_rate=sr)
hop_len = ms_to_samples(hparams.win_hop, sampling_rate=sr)

linear_spec, phase = linear_scale_spectrogram(wav, n_fft, hop_length=hop_len, win_length=win_len)
mel_spec, mfccs = mel_scale_spectrogram(wav, n_fft, sampling_rate=sr,
                                        hop_length=hop_len,
                                        win_length=win_len,
                                        n_mels=hparams.n_mels,
                                        fmin=hparams.mel_fmin,
                                        fmax=hparams.mel_fmax,
                                        n_mfcc=hparams.n_mfcc)

power_db = librosa.amplitude_to_db(linear_spec, ref=np.max(linear_spec))
mel_db = librosa.power_to_db(mel_spec, ref=np.max(mel_spec))

# TODO: I don't know if the min/max dB shown are correct.
# TODO: I don't know if the min/max frequency show are correct.
# plot_waveform(wav, sampling_rate=sr)
plot_spectrogram(power_db, sampling_rate=sr, hop_length=hop_len, y_axis='linear', title='Linear power spectrogram')
plot_spectrogram(mel_db, sampling_rate=sr, hop_length=hop_len, y_axis='mel', title='Mel power spectrogram')

# mfccs = sklearn.preprocessing.scale(mfccs, axis=1)

# librosa.display.specshow(mfccs, sr=sampling_rate, hop_length=hop_length, x_axis='time')
# plt.title('(manual) MFCC')
# plt.colorbar()
# plt.tight_layout()
# plt.show()