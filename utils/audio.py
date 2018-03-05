import librosa
import matplotlib.pyplot as plt
import numpy as np
from librosa import display

from hparams import hparams


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


def save_wav(wav_path, wav, sampling_rate=hparams.sampling_rate):
    """
    Write a WAV file to disk.

    :param wav_path: string
        Path to the file to write to.

    :param wav: np.ndarray [shape=(n,) or (2,n)]
        Audio time series to save.

    :param sampling_rate: int > 0 [scalar]
        Sampling rate of `wav`.
    """
    librosa.output.write_wav(wav_path, wav.astype(np.int16), sampling_rate)


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


def mel_scale_spectrogram(wav, n_fft, sampling_rate, n_mels, fmin, fmax, n_mfcc, hop_length, win_length, power):
    """
    Calculate a Mel-scaled spectrogram as well as Mel-frequency cepstral coefficients (MFCCs) from a signal.

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

    :param n_mfcc: int > 0 [scalar]
        Number of MFCCs to return.

    :param hop_length: int > 0 [scalar]
        Number of audio samples to hop between frames.

    :param win_length: int <= n_fft [scalar]
        Length of each frame in audio samples.

    :param power: float > 0 [scalar]
        Exponent for the magnitudes of the linear-scale spectrogram.
        e.g., 1 for energy, 2 for power, etc.

    :return:
        mel_spec: np.ndarray [shape=(1 + n_fft/2, t)]
            STFT matrix of the Mel-scaled spectrogram.

        mfccs: np.ndarray [shape=(n_mfcc, t)]
            Mel-frequency cepstral coefficients (MFCCs) for each frame.

    :notes:
        Setting `win_length` to `n_fft` results in exactly the same values the librosa build in functions
        `librosa.feature.mfcc` and `librosa.feature.melspectrogram` would deliver.
    """

    # TODO: Append shapes of the variables after each function call.

    # This implementation calculates the Mel-scaled spectrogram and the mfccs step by step.
    # Both `librosa.feature.mfcc` and `librosa.feature.melspectrogram` could be used to do this in fewer lines of code.
    # However, they do not allow to control the window length used in the initial stft calculation.
    # Setting `win_length` to `n_fft` results in exactly the same values the librosa build in functions would deliver.

    # Short-time Fourier transform of the signal to create a linear-scale spectrogram.
    mag_phase = librosa.stft(y=wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    # Extract the magnitudes from the linear-scale spectrogram.
    mag_spec = np.abs(mag_phase)

    # Raise the linear-scale spectrogram magnitudes to the power of `power`.
    # `power` = 1 for energy,`power` = 2 for power, etc.
    linear_spec = mag_spec ** power

    # Create a filter-bank matrix to combine FFT bins into Mel-frequency bins.
    mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=True)

    # Apply Mel-filters to create a Mel-scaled spectrogram.
    mel_spec = np.dot(mel_basis, linear_spec)
    mel_spec_db = librosa.power_to_db(mel_spec)

    # Calculate Mel-frequency cepstral coefficients (MFCCs) from the Mel-spectrogram.
    mfccs = librosa.feature.mfcc(S=mel_spec_db, sr=sampling_rate, n_mfcc=n_mfcc)

    return mel_spec, mfccs


def linear_scale_spectrogram(wav, n_fft, hop_length=None, win_length=None):
    mag_phase = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag, phase = librosa.magphase(mag_phase)

    return mag, phase


def plot_spectrogram(spec_db, sampling_rate, hop_length, title, y_axis='log'):
    librosa.display.specshow(spec_db, sr=sampling_rate, hop_length=hop_length, y_axis=y_axis, x_axis='time')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
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
