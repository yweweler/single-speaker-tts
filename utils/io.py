import librosa
import numpy as np

from hparams import hparams


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