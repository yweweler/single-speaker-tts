import librosa
import numpy as np

from audio.features import linear_scale_spectrogram
from audio.synthesis import spectrogram_to_wav


def pitch_shift(wav, sampling_rate, octaves):
    """
    Pitch-shift a waveform by `octaves` octaves.

    Arguments:
        wav (np.ndarray):
            Audio time series to pitch shift.
            The shape is expected to be shape=(n,) for an mono waveform.

        sampling_rate (:obj:`int`, optional):
            Target sampling rate. When None is used, the sampling rate inferred from the file is used.
            Defaults to None.

        octaves (float):
            Octaves to shift the pitch up or down.
            Each octave is divided into 12 half-steps.
            Therefore to shift one half-step one can use `octaves=1/12`.

    Returns:
        (np.ndarray):
            Audio time series.
            The shape of the returned array is shape=(n,) and the arrays dtype is np.float32.
    """
    rate = 2.0 ** (-octaves)

    # Stretch time.
    streched = time_stretch(wav, rate)

    # Resample the signal.
    y_shift = librosa.core.resample(streched, float(sampling_rate) / rate, sampling_rate)

    # Crop to the same dimension as the input.
    return librosa.util.fix_length(y_shift, len(wav))


def time_stretch(wav, rate):
    """
    Time-stretch an audio series by a fixed rate.

    Arguments:
        wav (np.ndarray):
            Audio time series to time stretch.
            The shape is expected to be shape=(n,) for an mono waveform.

        rate (float):
            Factor used to stretch the signal. `rate` is required to be > 0.
            With `rate` > 1.0 the signal is speed up.
            With 0.0 < `rate` < 1.0 the signal is slowed down.

    Returns:
        (np.ndarray):
            Audio time series.
            The shape of the returned array is shape=(n,) and the arrays dtype is np.float32.
    """
    if rate <= 0.0:
        raise ValueError('The fixed rate used to stretch the signal must be greater 0.')

    n_fft = 1024
    win_len = n_fft
    hop_len = win_len // 4
    reconstr_iters = 25

    # Construct the stft.
    stft = linear_scale_spectrogram(wav, n_fft, hop_len, win_len)

    # Stretch by phase vocoding.
    stft_stretch = librosa.core.phase_vocoder(stft, rate)

    # Get the magnitudes.
    mag = np.abs(stft_stretch)

    # Invert the stft.
    reconstr = spectrogram_to_wav(mag, win_len, hop_len, n_fft, reconstr_iters)

    return reconstr
