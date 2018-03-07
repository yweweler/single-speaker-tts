import librosa
import matplotlib.pyplot as plt
import numpy as np
import pysptk
import sklearn
from librosa import display

from hparams import hparams
from utils.audio import load_wav, plot_spectrogram, linear_scale_spectrogram, ms_to_samples, \
    mel_scale_spectrogram, calculate_mfccs, plot_feature_frames, calculate_mceps, plot_waveform, save_wav
from utils.synthesis import spectrogram_to_wav


def resynth_wav_using_mcep(wav, n_fft, hop_len, n_mceps, mcep_alpha):
    from pysptk.synthesis import MLSADF, Synthesizer

    # Calculate mceps.
    mceps, mc = calculate_mceps(wav, hop_len=hop_len,
                                n_mceps=n_mceps, alpha=mcep_alpha, n_fft=n_fft)

    # Calculate source excitation.
    pitch = pysptk.swipe(wav.astype(np.float64),
                         fs=sr, hopsize=hop_len, min=80, max=260, otype="pitch")

    source_excitation = pysptk.excite(pitch, hop_len)

    # print("pitch.shape", pitch.shape)
    # plt.plot(pitch, label="Source pitch")
    # plt.show()

    # print("source_excitation.shape", source_excitation.shape)
    # plt.plot(source_excitation, label="Source excitation")
    # plt.show()

    # Convert Mel-cepsrum to MLSA filter coefficients.
    b = pysptk.mc2b(mc, alpha=mcep_alpha)

    # Re-synthesize wav.
    synthesizer = Synthesizer(MLSADF(order=n_mceps, alpha=mcep_alpha), hop_len)
    wav_synth = synthesizer.synthesis(source_excitation, b)

    return mc, wav_synth


wav_path = '/home/yves-noel/documents/master/projects/datasets/timit/TIMIT/TRAIN/DR1/FCJF0/SA1.WAV'
n_fft = hparams.n_fft

wav, sr = load_wav(wav_path, sampling_rate=None)
print("min(wav) = {}".format(min(wav)))
print("max(wav) = {}".format(max(wav)))
print("type(wav) = {}".format(wav.dtype))

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

# wav to mcep to wav resynthesis.
# mc, wav_synth = resynth_wav_using_mcep(wav, hparams.n_fft, hop_len, 39, 0.42)
# save_wav('/tmp/original.wav', wav, sr, norm=True)
# save_wav('/tmp/synthesized.wav', wav_synth, sr, norm=True)
# plot_feature_frames(mc.T, sampling_rate=sr, hop_length=hop_len, title='MCEP')
# plot_waveform(wav, sampling_rate=sr)
# plot_waveform(wav_synth, sampling_rate=sr)
# exit()

wav_synth = spectrogram_to_wav(np.abs(linear_spec),
                               win_len,
                               hop_len,
                               hparams.n_fft,
                               50)
save_wav('/tmp/synthesized_griffin_v3.wav', wav_synth, sr, norm=True)
exit()

plot_spectrogram(linear_spec_db,
                 sampling_rate=sr,
                 hop_length=hop_len,
                 fmin=hparams.mel_fmin,
                 fmax=hparams.mel_fmax,
                 y_axis='linear',
                 title='Linear-scale power spectrogram')

# plot_spectrogram(mceps,
#                  sampling_rate=sr,
#                  hop_length=hop_len,
#                  fmin=hparams.mel_fmin,
#                  fmax=hparams.mel_fmax,
#                  y_axis='linear',
#                  title='Spectral envelope estimate from mel-cepstrum')

# plot_spectrogram(mel_spec_db,
#                  sampling_rate=sr,
#                  hop_length=hop_len,
#                  fmin=hparams.mel_fmin,
#                  fmax=hparams.mel_fmax,
#                  y_axis='mel',
#                  title='Mel-scale power spectrogram')

# # Calculate mfcc delta features.
# mfccs_d1 = librosa.feature.delta(mfccs, width=3, order=1)
# # Calculate mfcc delta-delta features.
# mfccs_d2 = librosa.feature.delta(mfccs, width=3, order=2)
#
# features = np.vstack((mfccs, mfccs_d1, mfccs_d2))
#
# plot_feature_frames(mfccs, sampling_rate=sr, hop_length=hop_len, title='MFCC')
# plot_feature_frames(mfccs_d1, sampling_rate=sr, hop_length=hop_len, title=r'MFCC-$\Delta$')
# plot_feature_frames(mfccs_d2, sampling_rate=sr, hop_length=hop_len, title=r'MFCC-$\Delta^2$')
