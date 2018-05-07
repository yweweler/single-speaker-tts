import librosa
import numpy as np

from audio.conversion import ms_to_samples
from audio.features import linear_scale_spectrogram
from audio.synthesis import spectrogram_to_wav


def pitch_shift(wav, sampling_rate, octaves):
    """
    Pitch-shift a waveform by `octaves` octaves.

    Arguments:
        wav (np.ndarray):
            Audio time series to pitch shift.
            The shape is expected to be shape=(n,) for an mono waveform.

        sampling_rate (int):
            Sampling rate of `wav`.

        octaves (float):
            Octaves to shift the pitch up or down.
            Each octave is divided into 12 half-steps.
            Therefore to shift one half-step one can use `octaves=1/12`.

    Returns:
        (np.ndarray):
            Audio time series.
            The shape of the returned array is shape=(n,) and the arrays dtype is np.float32.

    Notes:
        - This implementation is derived from the function `librosa.effects.pitch_shift`.
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
            The shape of the returned array is shape=(n * rate,) and the arrays dtype is np.float32.

    Notes:
        - This implementation is derived from the function `librosa.effects.time_stretch`.
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


def crop_silence_left(wav, sampling_rate, length_ms, safe_crop=True):
    """
    Crops a chunk of `length_ms` ms at the beginning of a waveform.

    Arguments:
        wav (np.ndarray):
            Audio time series to time stretch.
            The shape is expected to be shape=(n,) for an mono waveform.

        sampling_rate (int):
            Sampling rate of `wav`.

        length_ms (float):
            Length in ms of the audio chunk to crop.

        safe_crop (boolean):
            Flag that enables safe cropping of silence. If False, `wav` is cropped even if this
            would mean removing a non-silence region. If True, it is ensured that only silence
            is cropped and the non-silence region are left untouched.
            Default is True.

    Returns:
        (np.ndarray):
            Cropped audio time series.
            The shape of the returned array is shape=(n - n_cropped,) and the arrays dtype is
            np.float32.

        (int):
            Gives `n_cropped`, the number of sampled that were cropped.
            n_cropped = min(samples(length_ms), silence_len_left)
    """
    assert (length_ms > 0), 'Crop length must be greater 0.'

    samples = ms_to_samples(length_ms, sampling_rate)
    assert (samples < len(wav)), 'Crop length can not be greater than the total wav length.'

    # Ensure that only silence is cropped.
    if safe_crop:
        _, non_silence_region = trim_silence(wav)
        # Acquire the actual length of silence at the beginning.
        silence_len_left = non_silence_region[0]

        # Prevent accidental cropping of non-silence.
        samples = min(samples, silence_len_left)

    return wav[samples:], samples


def crop_silence_right(wav, sampling_rate, length_ms, safe_crop=True):
    """
    Crops a chunk of `length_ms` ms at the end of a waveform.

    Arguments:
        wav (np.ndarray):
            Audio time series to time stretch.
            The shape is expected to be shape=(n,) for an mono waveform.

        sampling_rate (int):
            Sampling rate of `wav`.

        length_ms (float):
            Length in ms of the audio chunk to crop.

        safe_crop (boolean):
            Flag that enables safe cropping of silence. If False, `wav` is cropped even if this
            would mean removing a non-silence region. If True, it is ensured that only silence
            is cropped and the non-silence region are left untouched.
            Default is True.

    Returns:
        (np.ndarray):
            Cropped audio time series.
            The shape of the returned array is shape=(n - n_cropped,) and the arrays dtype is
            np.float32.

        (int):
            Gives `n_cropped`, the number of sampled that were cropped.
            n_cropped = min(samples(length_ms), silence_len_right)

    """
    assert (length_ms > 0), 'Crop length must be greater 0.'

    samples = ms_to_samples(length_ms, sampling_rate)
    assert (samples < len(wav)), 'Crop length can not be greater than the total wav length.'

    # Ensure that only silence is cropped.
    if safe_crop:
        _, non_silence_region = trim_silence(wav)
        # Acquire the actual length of silence at the end.
        silence_len_right = len(wav) - non_silence_region[1]

        # Prevent accidental cropping of non-silence.
        samples = min(samples, silence_len_right)

    return wav[:-samples], samples


def trim_silence(wav, threshold_db=40, ref=np.max):
    """
    Trim leading and trailing silence from an audio signal.

    Arguments:
        wav (np.ndarray):
            Audio time series to pitch shift.
            The shape is expected to be shape=(n,) for an mono waveform.

        threshold_db (float):
            The threshold (in decibels) below reference to consider as silence.
            Default is 40 dB.

        ref:
            The reference power. By default, it uses `np.max` and compares to the peak power in
            the signal.

    Returns:
        (np.ndarray):
            Trimmed audio time series.
            The shape of the returned array is shape=(n',) and the arrays dtype is np.float32.

        (np.ndarray):
            Audio sample indices encapsulating to the non-silent audio region.
            The trimming operation conforms to trimmed_wav = wav[ indices[0] : indices[1] ]
            The indices shape is shape=(2,).
    """
    return librosa.effects.trim(wav, threshold_db, ref)


def trim_silence_spectrogram(mag_spec_db, threshold_db, ref=np.max):
    ref_trim_spec_db = ref(mag_spec_db, axis=0)

    print('ref_trim_spec_db', ref_trim_spec_db)

    non_silent = (ref_trim_spec_db > threshold_db)
    non_silent = np.array(non_silent, dtype=np.int32)
    print('non_silent', non_silent)

    nonzero = np.flatnonzero(non_silent)
    print('nonzero', nonzero)

    if len(nonzero) == 0:
        return None

    trim_start = np.min(nonzero)
    trim_end = np.max(nonzero)

    mag_spec_db = mag_spec_db[:, trim_start:trim_end]
    print('mag_spec_db', mag_spec_db.shape)

    return mag_spec_db
