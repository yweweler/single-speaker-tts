import librosa


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
    n_bins = 12
    return librosa.effects.pitch_shift(wav, sampling_rate, n_bins * octaves, n_bins)


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

    return librosa.effects.time_stretch(wav, rate)
