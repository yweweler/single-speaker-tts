import os

from matplotlib import rc
from matplotlib import ticker

from visualization.helpers import plot_scalar_data
from visualization.tensorboard import load_run, merge_runs_by_tag

def __plot_results_feature_comparasion_training(merged_runs, run_names, tag, base_settings):
    # Skip 1000 steps (Every 50 steps a summary was written)
    _skip = 1000 // 50

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
        'labels': ['{} loss {}'.format(tag, name) for name in run_names],
        'ylog': True,
        'raw': True,      # Do plot the raw data.
        'smooth': True,  # Do plot the moving average.
        'win_len': 12     # Window size used for moving average.
    }
    _settings.update(base_settings)

    return plot_scalar_data(merged_runs[tag]['step'], merged_runs[tag]['value'], _skip, _settings)
    pass

def __plot_results_feature_comparasion_evaluation(merged_runs, run_names, tag, base_settings):
    _settings = {
        'color': [
            "#6C8EBF",  # dark blue
            "#82B366",  # dark green
            "#9673A6",  # dark violet
            "#D79B00",  # dark orange
            "#B85450",  # dark red
        ],
        # Strings to show in the legend.
        'labels': ['{} loss {}'.format(tag, name) for name in run_names],
        'ylog': True,
        'raw': True,      # Do plot the raw data.
        'smooth': False,  # Do not plot the moving average.
        'win_len': 12     # Window size used for moving average.
    }
    _settings.update(base_settings)

    return plot_scalar_data(merged_runs[tag]['step'], merged_runs[tag]['value'], None, _settings)

def plot_results_feature_comparasion():
    # Base folder to load the data from.
    _base_folder = '/home/yves-noel/documents/master/thesis/project/visualization/data/blizzard/nancy/features/'

    # Runs to load for plotting.
    _runs = {
        '160-mels': dict(),
        '120-mels': dict(),
        # '80-mels': dict(),
        '40-mels': dict(),
        # '20-mels': dict()
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
        'figsize': (1.5 * 14.0 / 2.54, 2.0 * (7.7 / 2.54)),
        'xlabel': 'Steps',
        'ylabel': 'Loss',
    }

    # Global matplotlib style options.
    rc('font', **{'family': 'serif',
                  'serif': ['cmr10'],
                  'size': 13
                  })
    rc('text', usetex=True)

    # PLOT TRAIN DECODER LOSSES ===============================================================
    fig = __plot_results_feature_comparasion_training(merged_runs, _runs.keys(),
                                                        'train-decoder', _base_settings)
    fig.show()

    # PLOT TRAIN POST-PROCESSING LOSSES =======================================================
    fig = __plot_results_feature_comparasion_training(merged_runs, _runs.keys(),
                                                        'train-post', _base_settings)
    fig.show()

    # PLOT TRAIN TOTAL LOSSES =================================================================
    fig = __plot_results_feature_comparasion_training(merged_runs, _runs.keys(),
                                                        'train-total', _base_settings)
    fig.show()


    # PLOT EVALUATION DECODER LOSSES ===============================================================
    fig = __plot_results_feature_comparasion_evaluation(merged_runs, _runs.keys(),
                                                        'eval-decoder', _base_settings)
    fig.show()

    # PLOT EVALUATION POST-PROCESSING LOSSES =======================================================
    fig = __plot_results_feature_comparasion_evaluation(merged_runs, _runs.keys(),
                                                        'eval-post', _base_settings)
    fig.show()

    # PLOT EVALUATION TOTAL LOSSES =================================================================
    fig = __plot_results_feature_comparasion_evaluation(merged_runs, _runs.keys(),
                                                        'eval-total', _base_settings)
    fig.show()


def plot_results_blizzard_model():
    # Base folder to load the data from.
    _base_folder = '/home/yves-noel/documents/master/thesis/project/visualization/data/blizzard/nancy/features/'

    # Runs to load for plotting.
    run_name = '160-mels'

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
        'figsize': (1.5 * 14.0 / 2.54, (7.7 / 2.54)),
        'xlabel': 'Steps',
        'ylabel': 'Loss',
        'xaxis_formatter': ticker.FuncFormatter(
            lambda x, pos: '{:.0f}k'.format(x / 1000.0)
        ),
        'win_len': 24  # Window size used for moving average.
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
        'labels': ['loss', 'avg. loss'],
        'ylog': False,
        'raw': True,      # Do plot the raw data.
        'smooth': True,   # Do not plot the moving average.
    }
    _settings.update(_base_settings)
    fig = plot_scalar_data([run_data[tag]['step']],
                           [run_data[tag]['value']], None, _settings)
    fig.show()
    # fig.savefig("data/blizzard/nancy/loss_loss_decoder.pdf", bbox_inches='tight')

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
        'labels': ['loss', 'avg. loss'],
        'ylog': False,
        'raw': True,  # Do plot the raw data.
        'smooth': True,  # Do not plot the moving average.
    }
    _settings.update(_base_settings)
    fig = plot_scalar_data([run_data[tag]['step']],
                           [run_data[tag]['value']], None, _settings)
    fig.show()
    # fig.savefig("data/blizzard/nancy/loss_loss_post_processing.pdf", bbox_inches='tight')

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
        'labels': ['train loss', 'avg. train loss'],
        'ylog': False,
        'raw': True,  # Do plot the raw data.
        'smooth': True,  # Do not plot the moving average.
    }
    _settings.update(_base_settings)
    fig = plot_scalar_data([run_data[tag]['step']],
                           [run_data[tag]['value']], None, _settings)
    fig.show()
    # fig.savefig("data/blizzard/nancy/loss_loss_train.pdf", bbox_inches='tight')

    # ==============================================================================================
    # Blizzard Nancy Evaluate Loss (total)
    # ==============================================================================================
    tag = 'eval-total'
    _settings = {
        'color': [
            "#6C8EBF",  # dark blue
        ],
        'ylim': (0.2, 0.3),
        'yticks': [0.21, 0.22, 0.25],
        'labels': ['evaluate loss', 'avg. evaluate loss'],
        'ylog': True,
        'raw': True,  # Do plot the raw data.
        'smooth': False,  # Do not plot the moving average.
    }
    _settings.update(_base_settings)
    fig = plot_scalar_data([run_data[tag]['step']],
                           [run_data[tag]['value']], None, _settings)
    fig.show()
    # fig.savefig("data/blizzard/nancy/loss_loss_eval.pdf", bbox_inches='tight')

if __name__ == '__main__':
   plot_results_feature_comparasion()
   plot_results_blizzard_model()