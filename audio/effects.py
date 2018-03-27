import librosa


def pitch_shift(wav, sampling_rate, octaves):
    """
    Pitch-shift the waveform by `octaves` octaves.

    Arguments:
        wav (np.ndarray):
            Audio time series to save.
            The shape is expected to be shape=(n,) for an mono waveform.

        sampling_rate (:obj:`int`, optional):
            Target sampling rate. When None is used, the sampling rate inferred from the file is used.
            Defaults to None.

        octaves (float):
            Octaves to shift the pitch up or down.
            Each octave is divided into 12 half-steps.
            Therefore to shift one half-step one can use `octaves=1/12`.

    Returns:

    """
    n_bins = 12
    return librosa.effects.pitch_shift(wav, sampling_rate, n_bins * octaves, n_bins)
