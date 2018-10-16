import librosa
import numpy as np
from librosa import display
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter


def plot_spectrogram(spec_db,
                     sampling_rate,
                     hop_length,
                     fmin,
                     fmax,
                     y_axis,
                     title='',
                     figsize=(1.5 * 14.0 / 2.54, 7.7 / 2.54),
                     _formater=None):
    """
    Plots a spectrogram given a Short-time Fourier transform matrix.

    Arguments:
        spec_db (np.ndarray):
            STFT matrix of a spectrogram.
            The shape is expected to be shape=(n_bins, t) with
            n_bins being the number of frequency bins and t the number of frames.

        sampling_rate (int):
            Sampling rate of `spec_db`.

        hop_length (int):
            Number of audio samples to hop between frames.

        fmin (float):
            Lowest frequency (in Hz).

        fmax (float):
            Highest frequency (in Hz).

        y_axis (str):
            Range for the y-axes.

            Valid types are:

            - None, 'none', or 'off' : no axis decoration is displayed.
            - 'linear', 'fft', 'hz' : Frequency range is determined by
              the FFT window and sampling rate.
            - 'log' : The spectrum is displayed on a log scale.
            - 'mel' : Frequencies are determined by the mel scale.
            - 'cqt_hz' : Frequencies are determined by the CQT scale.
            - 'cqt_note' : Pitches are determined by the CQT scale.

        title (str):
            Title of the plot.
    """
    fig = plt.figure(figsize=figsize, dpi=100)
    librosa.display.specshow(spec_db,
                             # x_coords=np.arange(0, 150) + 160,
                             sr=sampling_rate,
                             hop_length=hop_length,
                             fmin=fmin,
                             fmax=fmax,
                             y_axis=y_axis,
                             x_axis='time')
    # plt.title(title)
    plt.set_cmap('viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel("Time (s)")

    if _formater is not None:
        ax = plt.gca()
        plt.ylabel('kHz')
        ax.yaxis.set_major_formatter(_formater)
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    plt.tight_layout()
    plt.show()

    return fig


def plot_feature_frames(features, sampling_rate, hop_length, title):
    """
    Plot a sequence of feature vectors that were computed for a framed signal.

    Arguments:
        features (np.ndarray):
            Feature time series to plot.
            The shape is expected to be shape=(n, t) with
            n being the number of features and t the number of feature frames.

        sampling_rate (int):
            Sampling rate of the original signal the features were computed on.

        hop_length (int):
            Number of audio samples that were hopped between frames.

        title (str):
            Title of the plot.
    """
    axes = librosa.display.specshow(data=features, sr=sampling_rate, hop_length=hop_length,
                                    x_axis='time')
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
    Plot a mono or stereo waveform signal in the time domain.

    Arguments:
        wav (np.ndarray):
            Audio time series.
            The shape is expected to be shape=(n,) for an mono waveform
            or shape(2, n) for an stereo waveform.

        sampling_rate (int):
            Sampling rate of `wav`.

        title (str):
            Title of the plot.
    """
    librosa.display.waveplot(wav, sr=sampling_rate)
    plt.title(title)
    plt.tight_layout()
    plt.show()
