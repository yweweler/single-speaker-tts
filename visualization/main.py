import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from visualization.helpers import moving_avg
from visualization.tensorboard import load_scalar


def __plot_scalar_data(steps, values, win_len=12, settings=None):
    _display = {
        'figsize': ((1.5 * 14.0 / 2.54) / 1.0, (7.7 / 2.54)),
        'color': ["#E1D5E7"],
        'color_smooth': ["#9673A6"],
        'title': 'unknown title',
        'xlabel': 'Steps',
        'ylabel': 'unknown',
        'ylim': None,
        'yticks': None,
        'xaxis_formatter': None,
        'yaxis_formatter': None,
    }

    if settings is not None:
        _display.update(settings)

    fig = plt.figure(figsize=_display['figsize'], dpi=100)

    for i, (subplot_steps, subplot_values) in enumerate(zip(steps, values)):
        # Plot the i'th subplots raw values.
        plt.semilogy(subplot_steps, subplot_values, color=_display['color'][i])
        # plt.plot(subplot_steps, subplot_values, color=_display['color'][i])

        # Smooth the i'th subplots values using a moving avg.
        values_smooth = moving_avg(subplot_values, win_len, 'same')

        # Discard values at the beginning and end where the window does not fit.
        steps_smoothed = subplot_steps[win_len - 1:-(win_len - 1)]
        values_smooth = values_smooth[win_len - 1:-(win_len - 1)]

        # Plot the i'th subplots smoothed values.
        plt.semilogy(steps_smoothed, values_smooth, color=_display['color_smooth'][i])
        # plt.plot(steps_smoothed, values_smooth, color=_display['color_smooth'][i])

    plt.grid(True, which='both', linestyle='dashed')
    plt.title(_display['title'])
    plt.xlabel(_display['xlabel'])
    plt.ylabel(_display['ylabel'])

    plt.xlim((0, max(map(np.max, steps))))

    if _display['ylim'] is not None:
        plt.ylim(_display['ylim'])

    ax = fig.gca()
    if _display['yticks'] is not None:
        ax.set_yticks(_display['yticks'])

    if _display['yaxis_formatter'] is not None:
        ax.yaxis.set_major_formatter(_display['yaxis_formatter'])

    if _display['xaxis_formatter'] is not None:
        ax.xaxis.set_major_formatter(_display['xaxis_formatter'])

    # Render grid above the lines.
    plt.rcParams['axes.axisbelow'] = False

    return fig


def plot_scalar(json_path, title, ylabel):
    data = load_scalar(json_path)
    return __plot_scalar_data(data['step'], data['value'], title, ylabel)


if __name__ == '__main__':
    train_loss_loss = load_scalar('data/blizzard/nancy/train-tag-loss_loss.json')
    train_loss_decoder = load_scalar('data/blizzard/nancy/train-tag-loss_loss_decoder.json')
    train_loss_post = load_scalar('data/blizzard/nancy/train-tag-loss_loss_post_processing.json')

    eval_loss_loss = load_scalar('data/blizzard/nancy/evaluate-tag-loss_loss.json')
    eval_loss_decoder = load_scalar('data/blizzard/nancy/evaluate-tag-loss_loss_decoder.json')
    eval_loss_post = load_scalar('data/blizzard/nancy/evaluate-tag-loss_loss_post_processing.json')

    print(min(train_loss_loss['value']))
    print(max(eval_loss_loss['value']))

    __plot_scalar_data([train_loss_loss['step']],
                       [train_loss_loss['value']],
                       settings={
                           'color': [
                               "#F8CECC",  # red
                           ],
                           'color_smooth': [
                               "#B85450",  # dark red
                           ],
                           'title': 'unknown title',
                           'xlabel': 'Steps',
                           'ylabel': 'unknown',
                       })
    plt.show()

    __plot_scalar_data([
        train_loss_loss['step'],
        eval_loss_loss['step']
    ], [
        train_loss_loss['value'],
        eval_loss_loss['value']
    ],
        settings={
            'figsize': (1.5 * 14.0 / 2.54, 2.0 * (7.7 / 2.54)),
            'color': [
                "#F8CECC",  # red
                "#DAE8FC",  # blue
            ],
            'color_smooth': [
                "#B85450",  # dark red
                "#6C8EBF",  # dark blue
            ],
            'title': 'Blizzard Nancy Train and Evaluate Loss',
            'xlabel': 'Steps',
            'ylabel': 'Loss',
            'ylim': (4e-2, 8e-1),
            'yticks': [4e-2, 7e-2, 1e-1, 2e-1, 8e-1],
            'yaxis_formatter': ticker.FuncFormatter(
                lambda y, pos: '{:.0e}'.format(y)
            ),
            'xaxis_formatter': ticker.FuncFormatter(
                lambda x, pos: '{:.0f}k'.format(x / 1000.0)
            )
        })
    plt.show()
