import librosa
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter


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