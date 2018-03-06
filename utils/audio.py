import librosa
import matplotlib.pyplot as plt
import numpy as np
import pysptk
from librosa import display
from matplotlib.ticker import ScalarFormatter

from hparams import hparams

# https://github.com/keithito/tacotron/blob/master/util/audio.py

# Griffin-Lim algorithm in Librosa:
# https://github.com/librosa/librosa/issues/434

# https://github.com/eYSIP-2017/eYSIP-2017_Speech_Spoofing_and_Verification

AUDIO_FLOAT_EPS = 1e-7


def power_to_decibel(power, ref, amin, max_db):
    """
    Convert a power spectrogram (amplitude squared) to decibel (dB).

    :param power: np.ndarray
        Power spectrum to convert.

    :param ref: scalar
        The amplitude `abs(power)` is scaled relative to `ref`:
        `10 * log10(power / ref)`.

    :param amin: float > 0 [scalar]
        Minimum threshold for `abs(power)` and `ref`.

    :param max_db: float >= 0 [scalar]
        Threshold the output at `max_db` below the peak:
        ``max(10 * log10(power)) - max_db``.

    :return:
        decibel: np.ndarray
        ``decibel ~= 10 * log10(power) - 10 * log10(ref)``
    """
    return 20 * np.log10(np.maximum(AUDIO_FLOAT_EPS, power))


def load_wav(wav_path, sampling_rate=hparams.sampling_rate, offset=0.0, duration=None):
    """
    Load an WAV file as a floating point time series from disk.

    :param wav_path: string
        Path of the WAV file.

    :param sampling_rate: number > 0 [scalar]
        Target sampling rate.

        'None' uses the native sampling rate.

    :param offset: float
        Offset to start loading the file at (in seconds).

    :param duration: float
        Only load up to this much audio (in seconds).

    :return:
        y: np.ndarray [shape=(n,) or (2, n)]
            Audio time series.

        sr: number > 0 [scalar]
            Sampling rate of `y`.
    """
    return librosa.core.load(wav_path, sr=sampling_rate, offset=offset, duration=duration)


def save_wav(wav_path, wav, sampling_rate=hparams.sampling_rate, norm=False):
    """
    Write a WAV file to disk.

    :param wav_path: string
        Path to the file to write to.

    :param wav: np.ndarray [shape=(n,) or (2,n)]
        Audio time series to save.

    :param sampling_rate: int > 0 [scalar]
        Sampling rate of `wav`.

    :param norm: boolean [scalar]
        Enable amplitude normalization.
        Scale the data to the range [-1, +1].
    """
    librosa.output.write_wav(wav_path, wav.astype(np.float32), sampling_rate, norm=norm)


def samples_to_ms(samples, sampling_rate):
    """
    Convert a duration in samples into milliseconds.

    :param samples: int > 0 [scalar]
        Samples to convert into milliseconds.

    :param sampling_rate: int > 0 [scalar]
        Sampling rate of `wav`.

    :return: float > 0 [scalar]
        Duration in ms.
    """
    return (samples / sampling_rate) * 1000


def ms_to_samples(ms, sampling_rate):
    """
    Convert a duration in milliseconds into samples.

    :param ms: float > 0 [scalar]
        Duration in ms.

    :param sampling_rate: int > 0 [scalar]
        Sampling rate of `wav`.

    :return: int > 0 [scalar]
        Duration in samples.
    """
    return int((ms / 1000) * sampling_rate)


def mel_scale_spectrogram(wav, n_fft, sampling_rate, n_mels, fmin, fmax, hop_length, win_length, power):
    """
    Calculate a Mel-scaled spectrogram from a signal.

    :param wav: np.ndarray [shape=(n,)]
        Audio time series.

    :param n_fft: int > 0 [scalar]
        FFT window size.

    :param sampling_rate: int > 0 [scalar]
        Sampling rate of `wav`.

    :param n_mels: int > 0 [scalar]
        Number of Mel bands to generate.

    :param fmin: float >= 0 [scalar]
        Lowest frequency (in Hz).

    :param fmax: float >= 0 [scalar]
        Highest frequency (in Hz).
        If `None`, use `fmax = sampling_rate / 2.0`.

    :param hop_length: int > 0 [scalar]
        Number of audio samples to hop between frames.

    :param win_length: int <= n_fft [scalar]
        Length of each frame in audio samples.

    :param power: float > 0 [scalar]
        Exponent for the magnitudes of the linear-scale spectrogram.
        e.g., 1 for energy, 2 for power, etc.

    :return:
        mel_spec: np.ndarray [shape=(n_mels, t)]
            STFT matrix of the Mel-scaled spectrogram.

    :notes:
        Setting `win_length` to `n_fft` results in exactly the same values the librosa build in functions
        `librosa.feature.mfcc` and `librosa.feature.melspectrogram` would deliver.
    """
    # This implementation calculates the Mel-scaled spectrogram and the mfccs step by step.
    # Both `librosa.feature.mfcc` and `librosa.feature.melspectrogram` could be used to do this in fewer lines of code.
    # However, they do not allow to control the window length used in the initial stft calculation.
    # Setting `win_length` to `n_fft` results in exactly the same values the librosa build in functions would deliver.

    # Short-time Fourier transform of the signal to create a linear-scale spectrogram.
    # Return shape: (n_fft/2 + 1, n_frames), with n_frames being floor(len(wav) / win_hop).
    mag_phase_spec = librosa.stft(y=wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    # Extract the magnitudes from the linear-scale spectrogram.
    # Return shape: (n_fft/2 + 1, n_frames).
    mag_spec = np.abs(mag_phase_spec)

    # Raise the linear-scale spectrogram magnitudes to the power of `power`.
    # `power` = 1 for energy,`power` = 2 for power, etc.
    # Return shape: (n_fft/2 + 1, n_frames).
    linear_spec = mag_spec ** power

    # Create a filter-bank matrix to combine FFT bins into Mel-frequency bins.
    # Return shape: (n_mels, n_fft/2 + 1).
    mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=True)

    # Apply Mel-filters to create a Mel-scaled spectrogram.
    # Return shape: (n_mels, n_frames).
    mel_spec = np.dot(mel_basis, linear_spec)

    return mel_spec


def calculate_mfccs(mel_spec, sampling_rate, n_mfcc):
    """
    Calculate Mel-frequency cepstral coefficients (MFCCs) from a Mel-scaled spectrogram.

    :param mel_spec: np.ndarray [shape=(n_mels, t)]
        STFT matrix of the Mel-scaled spectrogram.

    :param sampling_rate: int > 0 [scalar]
        Sampling rate of `wav`.

    :param n_mfcc: int > 0 [scalar]
        Number of MFCCs to return.

    :return:
        mfccs: np.ndarray [shape=(n_mfcc, t)]
            Mel-frequency cepstral coefficients (MFCCs) for each frame.
    """
    # Calculate Mel-frequency cepstral coefficients (MFCCs) from the Mel-spectrogram.
    # Return shape: (n_mfcc, n_frames).
    mfccs = librosa.feature.mfcc(S=mel_spec, sr=sampling_rate, n_mfcc=n_mfcc)

    return mfccs


def calculate_mceps(wav, hop_len, n_mceps, alpha, n_fft):
    win_len = n_fft
    frames = librosa.util.frame(wav, frame_length=win_len,
                                hop_length=hop_len).astype(np.float64).T
    frames *= pysptk.blackman(win_len)

    mc = pysptk.mcep(frames, n_mceps, alpha)
    logH = pysptk.mgc2sp(mc, alpha, 0.0, win_len).real

    print("logH mceps shape:", logH.T.shape)
    print("mc mceps shape:", mc.T.shape)
    return logH.T, mc


def linear_scale_spectrogram(wav, n_fft, hop_length=None, win_length=None):
    linear_spec = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    # mag, phase = librosa.magphase(linear_spec)

    return linear_spec


def plot_spectrogram(spec_db, sampling_rate, hop_length, fmin, fmax, y_axis, title):
    librosa.display.specshow(spec_db,
                             sr=sampling_rate,
                             hop_length=hop_length,
                             fmin=fmin,
                             fmax=fmax,
                             y_axis=y_axis,
                             x_axis='time')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()


def plot_feature_frames(features, sampling_rate, hop_length, title):
    """
    Plot a sequence of feature vectors that were computed for a framed signal.

    :param features: np.ndarray [shape=(n, t)]
        Feature time series to plot.

    :param sampling_rate: number > 0 [scalar]
        Sampling rate of the original signal the features were computed on.

    :param hop_length: int > 0 [scalar]
        Number of audio samples that were hopped between frames.

    :param title: string
        Title of the plot.
    """
    axes = librosa.display.specshow(features, sr=sampling_rate, hop_length=hop_length, x_axis='time')
    axes.yaxis.set_major_formatter(ScalarFormatter())
    axes.yaxis.set_ticks(np.arange(features.shape[0] + 1))
    plt.ylim(ymin=0, ymax=features.shape[0])

    plt.set_cmap('magma')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def plot_waveform(wav, sampling_rate, title='Mono'):
    """
    Plot a waveform signal.

    :param wav: np.ndarray [shape=(n,) or (2, n)]
        Audio time series to plot.

    :param sampling_rate: number > 0 [scalar]
        Sampling rate of `wav`.

    :param title: string
        Title of the plot.
    """
    librosa.display.waveplot(wav, sr=sampling_rate)
    plt.title(title)
    plt.tight_layout()
    plt.show()
