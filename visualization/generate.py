import os
import math

import numpy as np

from matplotlib import rc
from matplotlib import ticker

from audio.conversion import ms_to_samples, magnitude_to_decibel
from audio.features import linear_scale_spectrogram, mel_scale_spectrogram
from audio.io import load_wav
from audio.visualization import plot_spectrogram
from visualization.helpers import plot_scalar_data
from visualization.tensorboard import load_run, merge_runs_by_tag

def __plot_results_feature_comparasion_training(merged_runs, labels, tag, base_settings):
    # Skip 1000 steps (Every 50 steps a summary was written)
    _skip = 500 // 50

    _settings = {
        'color': [
            "#DAE8FC",  # blue
            "#D5E8D4",  # green
            "#E1D5E7",  # violet
            "#FFE6CC",  # orange
            "#F8CECC",  # red
        ],
        'color_smooth': [
            "#6C8EBF",  # dark blue
            "#82B366",  # dark green
            "#9673A6",  # dark violet
            "#D79B00",  # dark orange
            "#B85450",  # dark red
        ],
        # Strings to show in the legend.
        'labels': labels,
        'ylog': True,
        'raw': True,      # Do plot the raw data.
        'smooth': True,  # Do plot the moving average.
        'win_len': 50     # Window size used for moving average.
    }
    _settings.update(base_settings)

    return plot_scalar_data(merged_runs[tag]['step'], merged_runs[tag]['value'], _skip, _settings)

def __plot_results_feature_comparasion_evaluation(merged_runs, labels, tag, base_settings):
    # Skip 10000 steps (Every 5000 steps a summary was written)
    _skip = 10000 // 5000

    _settings = {
        'color': [
            "#6C8EBF",  # dark blue
            "#82B366",  # dark green
            "#9673A6",  # dark violet
            "#D79B00",  # dark orange
            "#B85450",  # dark red
        ],
        # Strings to show in the legend.
        'labels': labels,
        'ylog': True,
        'raw': True,      # Do plot the raw data.
        'smooth': False,  # Do not plot the moving average.
        'win_len': 12     # Window size used for moving average.
    }
    _settings.update(base_settings)

    return plot_scalar_data(merged_runs[tag]['step'], merged_runs[tag]['value'], _skip, _settings)

def plot_results_feature_comparasion():
    # Base folder to load the data from.
    _base_folder = 'data/blizzard/nancy/features/'

    # Runs to load for plotting.
    _runs = {
        '160-mels': dict(),
        '120-mels': dict(),
        '80-mels': dict(),
        '40-mels': dict(),
        '20-mels': dict()
    }
    # Max. number of steps to load.
    _steps = 200000

    # Individual tags to prepare for plotting.
    _tags = [
        'eval-total',
        'eval-decoder',
        'eval-post',
        'train-total',
        'train-decoder',
        'train-post'
    ]

    # Load the data for all runs.
    for run in _runs:
        # Get the folder for the current run.
        _run_folder = os.path.join(_base_folder, run)
        # Load the run.
        _run_data = load_run(folder=_run_folder, steps=_steps)
        _runs[run] = _run_data

    # Merge the runs by stacking their tags.
    merged_runs = merge_runs_by_tag(runs=_runs, tags=_tags)

    # Shared base settings for plotting.
    _base_settings = {
        'figsize': (1.5 * 14.0 / 2.54, 3.0 * (7.7 / 2.54)),
        'xlabel': 'Steps',
        'ylabel': 'Loss'
    }

    # Global matplotlib style options.
    rc('font', **{'family': 'serif',
                  'serif': ['cmr10'],
                  'size': 13
                  })
    rc('text', usetex=True)

    _base_settings.update({
        'ylim': (0.02, 0.07),
        'yticks': [0.05, 0.07],
    })

    # PLOT TRAIN DECODER LOSSES ===============================================================
    tag = 'train-decoder'
    labels = ['train $\ell_{{d}}$ ({})'.format(name) for name in _runs.keys()]
    fig = __plot_results_feature_comparasion_training(merged_runs, labels, tag, _base_settings)
    fig.show()
    fig.savefig("data/blizzard/nancy/features/train-loss-decoder.pdf", bbox_inches='tight')

    _base_settings.update({
        'ylim': (0.02, 0.07),
        'yticks': [0.05],
    })

    # PLOT TRAIN POST-PROCESSING LOSSES =======================================================
    tag = 'train-post'
    labels = ['train $\ell_{{p}}$ ({})'.format(name) for name in _runs.keys()]
    fig = __plot_results_feature_comparasion_training(merged_runs, labels, tag, _base_settings)
    fig.show()
    fig.savefig("data/blizzard/nancy/features/train-loss-post-processing.pdf", bbox_inches='tight')

    _base_settings.update({
        'ylim': (0.05, 0.13),
        'yticks': [0.05, 0.07, 0.08, 0.09, 0.1, 0.13],
    })

    # PLOT TRAIN TOTAL LOSSES =================================================================
    tag = 'train-total'
    labels = ['train $\ell_{{t}}$ ({})'.format(name) for name in _runs.keys()]
    fig = __plot_results_feature_comparasion_training(merged_runs, labels, tag, _base_settings)
    fig.show()
    #fig.savefig("data/blizzard/nancy/features/train-loss-total.pdf", bbox_inches='tight')

    _base_settings.update({
        'figsize': (1.5 * 14.0 / 2.54, 2.0 * (7.7 / 2.54)),
        'ylim': (0.1, 0.2),
        'yticks': [0.105, 0.11, 0.12],
    })

    # PLOT EVALUATION DECODER LOSSES ===============================================================
    tag = 'eval-decoder'
    labels = ['evaluate $\ell_{{d}}$ ({})'.format(name) for name in _runs.keys()]
    fig = __plot_results_feature_comparasion_evaluation(merged_runs, labels, tag, _base_settings)
    fig.show()
    #fig.savefig("data/blizzard/nancy/features/evaluate-loss-decoder.pdf", bbox_inches='tight')

    _base_settings.update({
        'ylim': (0.09, 0.2),
        'yticks': [0.095, 0.10, 0.11],
    })

    # PLOT EVALUATION POST-PROCESSING LOSSES =======================================================
    tag = 'eval-post'
    labels = ['evaluate $\ell_{{p}}$ ({})'.format(name) for name in _runs.keys()]
    fig = __plot_results_feature_comparasion_evaluation(merged_runs, labels, tag, _base_settings)
    fig.show()
    #fig.savefig("data/blizzard/nancy/features/evaluate-loss-post-processing.pdf",
    #            bbox_inches='tight')

    _base_settings.update({
        'ylim': (0.19, 0.4),
        'yticks': [0.19, 0.22, 0.25],
    })
    # PLOT EVALUATION TOTAL LOSSES =================================================================
    tag = 'eval-total'
    labels = ['evaluate $\ell_{{t}}$ ({})'.format(name) for name in _runs.keys()]
    fig = __plot_results_feature_comparasion_evaluation(merged_runs, labels, tag, _base_settings)
    fig.show()
    #fig.savefig("data/blizzard/nancy/features/evaluate-loss-total.pdf", bbox_inches='tight')


def plot_results_blizzard_model():
    # Base folder to load the data from.
    _base_folder = 'data/blizzard/nancy/features/'

    # Runs to load for plotting.
    run_name = '80-mels'

    # Max. number of steps to load.
    _steps = 500000

    # Individual tags to prepare for plotting.
    # _tags = [
    #     'eval-total',
    #     'train-total',
    #     'train-decoder',
    #     'train-post'
    # ]

    # Get the folder for the current run.
    _run_folder = os.path.join(_base_folder, run_name)
    # Load the run.
    run_data = load_run(folder=_run_folder, steps=_steps)

    # Shared base settings for plotting.
    _base_settings = {
        'figsize': (1.25 * 14.0 / 2.54, (7.7 / 2.54)),
        'xlabel': 'Steps',
        'ylabel': 'Loss',
        'xaxis_formatter': ticker.FuncFormatter(
            lambda x, pos: '{:.0f}k'.format(x / 1000.0)
        ),
        'win_len': 50  # Window size used for moving average.
    }

    # Global matplotlib style options.
    rc('font', **{'family': 'serif',
                  'serif': ['cmr10'],
                  'size': 13
                  })
    rc('text', usetex=True)

    # ==============================================================================================
    # Blizzard Nancy Train Loss (decoder)
    # ==============================================================================================
    tag = 'train-decoder'
    _settings = {
        'color': [
            "#F8CECC",  # red
        ],
        'color_smooth': [
            "#B85450",  # dark red
        ],
        'ylim': (2e-2, 8e-2),
        'yticks': [2e-2, 3e-2, 4e-2, 8e-2],
        'labels': ['train $\ell_{d}$', 'avg. train $\ell_{d}$'],
        'ylog': False,
        'raw': True,      # Do plot the raw data.
        'smooth': True,   # Do not plot the moving average.
    }
    _settings.update(_base_settings)
    fig = plot_scalar_data([run_data[tag]['step']],
                           [run_data[tag]['value']], None, _settings)
    fig.show()
    # fig.savefig("data/blizzard/nancy/survey/loss_loss_decoder.pdf", bbox_inches='tight')

    # ==============================================================================================
    # Blizzard Nancy Train Loss (post-processing)
    # ==============================================================================================
    tag = 'train-post'
    _settings = {
        'color': [
            "#F8CECC",  # red
        ],
        'color_smooth': [
            "#B85450",  # dark red
        ],
        'ylim': (2.0e-2, 6e-2),
        'yticks': [2.0e-2, 3e-2, 4e-2, 6e-2],
        'labels': ['train $\ell_{p}$', 'avg. train $\ell_{p}$'],
        'ylog': False,
        'raw': True,  # Do plot the raw data.
        'smooth': True,  # Do not plot the moving average.
    }
    _settings.update(_base_settings)
    fig = plot_scalar_data([run_data[tag]['step']],
                           [run_data[tag]['value']], None, _settings)
    fig.show()
    # fig.savefig("data/blizzard/nancy/survey/loss_loss_post_processing.pdf", bbox_inches='tight')

    # ==============================================================================================
    # Blizzard Nancy Train Loss (total)
    # ==============================================================================================
    tag = 'train-total'
    _settings = {
        'color': [
            "#F8CECC",  # red
        ],
        'color_smooth': [
            "#B85450",  # dark red
        ],
        'ylim': (4.0e-2, 1.5e-1),
        'yticks': [0.05, 0.07, 0.08, 0.1, 0.15],
        'labels': ['train $\ell_{t}$', 'avg. train $\ell_{t}$'],
        'ylog': False,
        'raw': True,  # Do plot the raw data.
        'smooth': True,  # Do not plot the moving average.
    }
    _settings.update(_base_settings)
    fig = plot_scalar_data([run_data[tag]['step']],
                           [run_data[tag]['value']], None, _settings)
    fig.show()
    # fig.savefig("data/blizzard/nancy/survey/loss_loss_train.pdf", bbox_inches='tight')

    # ==============================================================================================
    # Blizzard Nancy Evaluate Loss (total)
    # ==============================================================================================
    tag = 'eval-total'
    _settings = {
        'color': [
            "#6C8EBF",  # dark blue
        ],
        'ylim': (0.2, 0.3),
        'yticks': [0.2, 0.21, 0.22, 0.25, 0.3],
        'labels': ['evaluate $\ell_{t}$', 'avg. evaluate $\ell_{t}$'],
        'ylog': False,
        'raw': True,  # Do plot the raw data.
        'smooth': False,  # Do not plot the moving average.
    }
    _settings.update(_base_settings)
    fig = plot_scalar_data([run_data[tag]['step']],
                           [run_data[tag]['value']], None, _settings)
    fig.show()
    fig.savefig("data/blizzard/nancy/survey/loss_loss_eval.pdf", bbox_inches='tight')

def plot_results_ljspeech_model():
    # Base folder to load the data from.
    _base_folder = 'data/ljspeech/v1.1/features/'

    # Runs to load for plotting.
    run_name = '80-mels'

    # Max. number of steps to load.
    _steps = 500000

    # Individual tags to prepare for plotting.
    # _tags = [
    #     'eval-total',
    #     'train-total',
    #     'train-decoder',
    #     'train-post'
    # ]

    # Get the folder for the current run.
    _run_folder = os.path.join(_base_folder, run_name)
    # Load the run.
    run_data = load_run(folder=_run_folder, steps=_steps)

    # Shared base settings for plotting.
    _base_settings = {
        'figsize': (1.25 * 14.0 / 2.54, (7.7 / 2.54)),
        'xlabel': 'Steps',
        'ylabel': 'Loss',
        'xaxis_formatter': ticker.FuncFormatter(
            lambda x, pos: '{:.0f}k'.format(x / 1000.0)
        ),
        'win_len': 50  # Window size used for moving average.
    }

    # Global matplotlib style options.
    rc('font', **{'family': 'serif',
                  'serif': ['cmr10'],
                  'size': 13
                  })
    rc('text', usetex=True)

    # ==============================================================================================
    # Blizzard Nancy Train Loss (decoder)
    # ==============================================================================================
    tag = 'train-decoder'
    _settings = {
        'color': [
            "#F8CECC",  # red
        ],
        'color_smooth': [
            "#B85450",  # dark red
        ],
        'ylim': (2e-2, 8e-2),
        'yticks': [2e-2, 3e-2, 4e-2, 8e-2],
        'labels': ['train $\ell_{d}$', 'avg. train $\ell_{d}$'],
        'ylog': False,
        'raw': True,      # Do plot the raw data.
        'smooth': True,   # Do not plot the moving average.
    }
    _settings.update(_base_settings)
    fig = plot_scalar_data([run_data[tag]['step']],
                           [run_data[tag]['value']], None, _settings)
    fig.show()
    # fig.savefig("data/ljspeech/v1.1/survey/loss_loss_decoder.pdf", bbox_inches='tight')

    # ==============================================================================================
    # Blizzard Nancy Train Loss (post-processing)
    # ==============================================================================================
    tag = 'train-post'
    _settings = {
        'color': [
            "#F8CECC",  # red
        ],
        'color_smooth': [
            "#B85450",  # dark red
        ],
        'ylim': (2.0e-2, 6e-2),
        'yticks': [2.0e-2, 3e-2, 4e-2, 6e-2],
        'labels': ['train $\ell_{p}$', 'avg. train $\ell_{p}$'],
        'ylog': False,
        'raw': True,  # Do plot the raw data.
        'smooth': True,  # Do not plot the moving average.
    }
    _settings.update(_base_settings)
    fig = plot_scalar_data([run_data[tag]['step']],
                           [run_data[tag]['value']], None, _settings)
    fig.show()
    # fig.savefig("data/ljspeech/v1.1/survey/loss_loss_post_processing.pdf", bbox_inches='tight')

    # ==============================================================================================
    # Blizzard Nancy Train Loss (total)
    # ==============================================================================================
    tag = 'train-total'
    _settings = {
        'color': [
            "#F8CECC",  # red
        ],
        'color_smooth': [
            "#B85450",  # dark red
        ],
        'ylim': (4.0e-2, 1.5e-1),
        'yticks': [0.05, 0.07, 0.08, 0.1, 0.15],
        'labels': ['train $\ell_{t}$', 'avg. train $\ell_{t}$'],
        'ylog': False,
        'raw': True,  # Do plot the raw data.
        'smooth': True,  # Do not plot the moving average.
    }
    _settings.update(_base_settings)
    fig = plot_scalar_data([run_data[tag]['step']],
                           [run_data[tag]['value']], None, _settings)
    fig.show()
    # fig.savefig("data/ljspeech/v1.1/survey/loss_loss_train.pdf", bbox_inches='tight')

    # ==============================================================================================
    # Blizzard Nancy Evaluate Loss (total)
    # ==============================================================================================
    tag = 'eval-total'
    _settings = {
        'color': [
            "#6C8EBF",  # dark blue
        ],
        'ylim': (2e-1, 0.4),
        'yticks': [0.2, 0.23, 0.24, 0.3, 0.35, 0.4],
        'labels': ['evaluate $\ell_{t}$', 'avg. evaluate $\ell_{t}$'],
        'ylog': False,
        'raw': True,  # Do plot the raw data.
        'smooth': False,  # Do not plot the moving average.
    }
    _settings.update(_base_settings)
    fig = plot_scalar_data([run_data[tag]['step']],
                           [run_data[tag]['value']], None, _settings)
    fig.show()
    fig.savefig("data/ljspeech/v1.1/survey/loss_loss_eval.pdf", bbox_inches='tight')


def plot_liner_mel_spec_comparasion():
    ms_win_len = 50.0
    ms_win_hop = 12.5
    n_fft = 1024
    wav_path = '/home/yves-noel/documents/master/thesis/datasets/blizzard_nancy/wav/RURAL-02198.wav'

    wav, sr = load_wav(wav_path)
    win_len = ms_to_samples(ms_win_len, sampling_rate=sr)
    hop_len = ms_to_samples(ms_win_hop, sampling_rate=sr)

    linear_spec = linear_scale_spectrogram(wav, n_fft, hop_len, win_len).T

    mel_spec = mel_scale_spectrogram(wav,
                                     n_fft=n_fft,
                                     sampling_rate=sr,
                                     n_mels=80,
                                     fmin=0,
                                     fmax=sr // 2,
                                     hop_length=hop_len,
                                     win_length=win_len,
                                     power=1).T

    # ==================================================================================================
    # Convert the linear spectrogram into decibel representation.
    # ==================================================================================================
    linear_mag = np.abs(linear_spec)
    linear_mag_db = magnitude_to_decibel(linear_mag)

    # ==================================================================================================
    # Convert the mel spectrogram into decibel representation.
    # ==================================================================================================
    mel_mag = np.abs(mel_spec)
    mel_mag_db = magnitude_to_decibel(mel_mag)

    rc('font', **{'family': 'serif',
                  'serif': ['Computer Modern'],
                  'size': 13})
    rc('text', usetex=True)

    y_formater = ticker.FuncFormatter(
        lambda x, pos: '{:.0f}'.format(x / 1000.0)
    )

    linear_mag_db = linear_mag_db[int((0.20 * sr) / hop_len):int((1.85 * sr) / hop_len), :]
    fig = plot_spectrogram(linear_mag_db.T, sr, hop_len, 0.0, sr // 2.0, 'linear',
                            figsize=((1.0 / 1.35) * (14.0 / 2.54), 7.7 / 2.54),
                           _formater=y_formater)

    fig.savefig("/tmp/linear_spectrogram_raw_mag_db.pdf", bbox_inches='tight')

    def __tmp_fmt(x):
        if x == 0.0:
            return '{:.0f}'.format(x / 1000.0)
        elif x < 1000:
            return '{:.1f}'.format(x / 1000.0)
        else:
            return '{:.0f}'.format(math.floor(x / 1000.0))

    y_formater = ticker.FuncFormatter(
        lambda x, pos: __tmp_fmt(x)
    )

    mel_mag_db = mel_mag_db[int((0.20 * sr) / hop_len):int((1.85 * sr) / hop_len), :]
    fig = plot_spectrogram(mel_mag_db.T, sr, hop_len, 0.0, sr // 2.0, 'mel',
                            figsize=((1.025 / 1.35) * (14.0 / 2.54), 7.7 / 2.54),
                           _formater=y_formater)

    fig.savefig("/tmp/mel_spectrogram_raw_mag_db.pdf", bbox_inches='tight')


if __name__ == '__main__':
   plot_results_feature_comparasion()
   #plot_results_blizzard_model()
   # plot_results_ljspeech_model()
   # plot_liner_mel_spec_comparasion()
   pass