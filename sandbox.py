import librosa
import numpy as np
import pysptk

from hparams import hparams
from utils.conversion import ms_to_samples
from utils.features import linear_scale_spectrogram, mel_scale_spectrogram, calculate_mfccs, calculate_mceps
from utils.io import load_wav
from utils.visualization import plot_spectrogram, plot_feature_frames


def resynth_wav_using_mcep(wav, hop_len, n_mceps, mcep_alpha):
    from pysptk.synthesis import MLSADF, Synthesizer

    # Calculate mceps.
    mceps, mc = calculate_mceps(wav, n_fft=hparams.n_fft, hop_length=hop_len,
                                n_mceps=n_mceps, alpha=mcep_alpha)

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

    # save_wav('/tmp/synthesized.wav', wav_synth, sr, norm=True)
    # plot_feature_frames(mc.T, sampling_rate=sr, hop_length=hop_len, title='MCEP')

    # plot_spectrogram(mc,
    #                  sampling_rate=sr,
    #                  hop_length=hop_len,
    #                  fmin=hparams.mel_fmin,
    #                  fmax=hparams.mel_fmax,
    #                  y_axis='linear',
    #                  title='Spectral envelope estimate from mel-cepstrum')

    return mc, wav_synth


def calculate_mfccs_and_deltas(wav, hop_len, win_len):
    mel_spec = mel_scale_spectrogram(wav,
                                     n_fft=hparams.n_fft,
                                     sampling_rate=hparams.sampling_rate,
                                     hop_length=hop_len,
                                     win_length=win_len,
                                     n_mels=hparams.n_mels,
                                     fmin=hparams.mel_fmin,
                                     fmax=hparams.mel_fmax,
                                     power=2)

    # Convert the Mel-scaled spectrogram to its decibel representation.
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Calculate mfcc features.
    mfccs = calculate_mfccs(mel_spec_db, sr, hparams.n_mfcc)

    plot_spectrogram(mel_spec_db,
                     sampling_rate=hparams.sampling_rate,
                     hop_length=hop_len,
                     fmin=hparams.mel_fmin,
                     fmax=hparams.mel_fmax,
                     y_axis='mel',
                     title='Mel-scale power spectrogram')

    # # Calculate mfcc delta features.
    mfccs_d1 = librosa.feature.delta(mfccs, width=3, order=1)
    # Calculate mfcc delta-delta features.
    mfccs_d2 = librosa.feature.delta(mfccs, width=3, order=2)

    features = np.vstack((mfccs, mfccs_d1, mfccs_d2))

    plot_feature_frames(mfccs, sampling_rate=sr, hop_length=hop_len, title='MFCC')
    plot_feature_frames(mfccs_d1, sampling_rate=sr, hop_length=hop_len, title=r'MFCC-$\Delta$')
    plot_feature_frames(mfccs_d2, sampling_rate=sr, hop_length=hop_len, title=r'MFCC-$\Delta^2$')


def calculate_linear_spec(wav, hop_len, win_len):
    linear_spec = linear_scale_spectrogram(wav,
                                           n_fft=hparams.n_fft,
                                           hop_length=hop_len,
                                           win_length=win_len)

    # Convert the linear-scaled spectrogram to its decibel representation.
    linear_spec_db = librosa.amplitude_to_db(linear_spec, ref=np.max)

    plot_spectrogram(linear_spec_db,
                     sampling_rate=hparams.sampling_rate,
                     hop_length=hop_len,
                     fmin=hparams.mel_fmin,
                     fmax=hparams.mel_fmax,
                     y_axis='linear',
                     title='Linear-scale power spectrogram')


wav_path = '/home/yves-noel/documents/master/projects/datasets/timit/TIMIT/TRAIN/DR1/FCJF0/SA1.WAV'
wav, sr = load_wav(wav_path)

win_len = ms_to_samples(hparams.win_len, sampling_rate=sr)
hop_len = ms_to_samples(hparams.win_hop, sampling_rate=sr)

# calculate_linear_spec(wav, hop_len, win_len)
# calculate_mfccs_and_deltas(wav, hop_len, win_len)
# resynth_wav_using_mcep(wav, hop_len, 25, 0.35)
