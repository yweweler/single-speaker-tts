import librosa
import numpy as np

from hparams import hparams


def load_wav(wav_path, sampling_rate=None, offset=0.0, duration=None):
    """
    Load an WAV file as a floating point time series from disk.

    Arguments:
        wav_path (str):
            Path of the WAV file.

        sampling_rate (:obj:`int`, optional):
            Target sampling rate. When None is used, the sampling rate inferred from the file is used.
            Defaults to None.

        offset (:obj:`float`, optional):
            Offset to start loading the file at (in seconds).
            Defaults to 0.0.

        duration (:obj:`float`, optional):
            Only load up to this much audio (in seconds). When None is used,
            the file is loaded from `offset` to the end.
            Defaults to None.

    Returns:
        (np.ndarray, int):
            A tuple consisting of the audio time series and the sampling rate used for loading.
    """
    return librosa.core.load(wav_path, sr=sampling_rate, offset=offset, duration=duration)


def save_wav(wav_path, wav, sampling_rate=hparams.sampling_rate, norm=False):
    """
    Write a WAV file to disk.

    Arguments:
        wav_path (str):
            Path to the file to write to.

        wav (np.ndarray):
            Audio time series to save.
            The shape is expected to be shape=(n,) for an mono waveform
            or shape(2, n) for an stereo waveform.

        sampling_rate (int):
            Sampling rate of `wav`.

        norm (:obj:`bool`, optional):
            Enable amplitude normalization.
            For floating point `wav`, scale the data to the range [-1, +1].
    """
    librosa.output.write_wav(wav_path, wav.astype(np.float32), sampling_rate, norm=norm)
