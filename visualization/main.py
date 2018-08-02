import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from visualization.helpers import moving_avg
from visualization.tensorboard import load_scalar, load_attention_alignments


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
        'labels': ['unknown_{}'.format(i + 1) for i in range(len(values) * 2)]
    }

    if settings is not None:
        _display.update(settings)

    fig = plt.figure(figsize=_display['figsize'], dpi=100)

    for i, (subplot_steps, subplot_values) in enumerate(zip(steps, values)):
        # Plot the i'th subplots raw values.
        plt.semilogy(subplot_steps,
                     subplot_values,
                     color=_display['color'][i],
                     label=_display['labels'][i * 2 + 0])
        # Smooth the i'th subplots values using a moving avg.
        values_smooth = moving_avg(subplot_values, win_len, 'same')

        # Discard values at the beginning and end where the window does not fit.
        steps_smoothed = subplot_steps[win_len - 1:-(win_len - 1)]
        values_smooth = values_smooth[win_len - 1:-(win_len - 1)]

        # Plot the i'th subplots smoothed values.
        plt.semilogy(steps_smoothed,
                     values_smooth,
                     color=_display['color_smooth'][i],
                     label=_display['labels'][i * 2 + 1])
        # plt.plot(steps_smoothed, values_smooth, color=_display['color_smooth'][i])

    plt.legend(labels=_display['labels'])
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
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    if _display['xaxis_formatter'] is not None:
        ax.xaxis.set_major_formatter(_display['xaxis_formatter'])

    # Render grid above the lines.
    ax.set_axisbelow(False)

    return fig


def __plot_alignments(alignments, settings=None):
    _display = {
        'figsize': None,
        'title': 'unknown title',
        'xlabel': 'Decoder states',
        'ylabel': 'Encoder states',
    }

    if settings is not None:
        _display.update(settings)

    fig = plt.figure(figsize=_display['figsize'], dpi=100)

    # Remove unused dimension.
    alignments = alignments.squeeze(axis=0)
    # alignments = np.flip(alignments, axis=0)
    # alignments = alignments[:, :125]

    img = plt.imshow(alignments, interpolation='nearest', cmap='viridis')

    # Set the color bar tick step.
    step = 0.1
    fig.colorbar(img, ticks=np.arange(0, 1.0 / step) * step, orientation='horizontal')

    plt.title(_display['title'])
    plt.xlabel(_display['xlabel'])
    plt.ylabel(_display['ylabel'])

    # Flip the y axis for better readability.
    plt.gca().invert_yaxis()

    return fig


def __plot_alignment_progress(steps, alignments):
    _columns = 2

    n_alignments = len(alignments)

    n_rows = int(math.ceil(n_alignments / _columns))
    n_cols = int(_columns)

    fig, axes = plt.subplots(figsize=(
            (1.5 * 14.0 / 2.54) * 1.3,
            (1.5 * 14.0 / 2.54) * 1.3
        ),
        nrows=n_rows,
        ncols=n_cols,
        sharex='none',
        sharey='none'
    )

    # Deactivate unused subplot.
    for ax in axes.flat[-n_alignments:]:
        ax.axis('off')

    for i, ax in enumerate(axes.flat[:n_alignments]):
        step = steps[i]
        alignment = alignments[i]

        # Remove unused dimension and flip the array.
        alignment = alignment.squeeze(axis=0)

        # Flip yaxis in order to make alignments readable.
        # alignment = np.flip(alignment, axis=0)

        # Drop empty frames.
        alignment = alignment[:, :120]

        ax.set_title('{} steps'.format(step))
        img = ax.imshow(alignment, interpolation='nearest', cmap='viridis')

        # Set the color bar tick step.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.075)
        plt.colorbar(img, cax=cax)

        # Flip the y axis for better readability.
        ax.invert_yaxis()
        plt.setp(ax.get_xticklabels(), visible=True)

        # Activate a separate axis for each subplot.
        ax.axis('on')

    return fig


if __name__ == '__main__':
    train_loss_loss = load_scalar('data/blizzard/nancy/train-tag-loss_loss.json')
    train_loss_decoder = load_scalar('data/blizzard/nancy/train-tag-loss_loss_decoder.json')
    train_loss_post = load_scalar('data/blizzard/nancy/train-tag-loss_loss_post_processing.json')

    eval_loss_loss = load_scalar('data/blizzard/nancy/evaluate-tag-loss_loss.json')
    eval_loss_decoder = load_scalar('data/blizzard/nancy/evaluate-tag-loss_loss_decoder.json')
    eval_loss_post = load_scalar('data/blizzard/nancy/evaluate-tag-loss_loss_post_processing.json')

    alignments_base_path = 'data/blizzard/nancy/alignments'

    alignment_progress = [
        'alignments-1.npz',
            # 'alignments-101.npz',
            # 'alignments-201.npz',
        # 'alignments-501.npz',
        # 'alignments-1001.npz',
        # 'alignments-1501.npz',
        # 'alignments-2001.npz',
        # 'alignments-2501.npz',
        # 'alignments-3001.npz',
        # 'alignments-3501.npz',
        # 'alignments-4001.npz',
        # 'alignments-4501.npz',
        # 'alignments-5001.npz',
        # 'alignments-5501.npz',
        # 'alignments-6001.npz',
        # 'alignments-6501.npz',
        # 'alignments-7001.npz',
        # 'alignments-7501.npz',
        # 'alignments-8001.npz',
        # 'alignments-8501.npz',
        # 'alignments-9001.npz',
        # 'alignments-9501.npz',
        # 'alignments-10001.npz',
        # 'alignments-10501.npz',
        # 'alignments-11001.npz',
        'alignments-11501.npz',
        # 'alignments-12001.npz',
        # 'alignments-12501.npz',
        'alignments-13001.npz',
        # 'alignments-13501.npz',
        'alignments-14001.npz',
        # 'alignments-14501.npz',
        'alignments-15001.npz',
    ]

    # ==============================================================================================
    # A single attention alignment history plot.
    # ==============================================================================================
    alignments = load_attention_alignments('data/blizzard/nancy/alignments.npz')
    fig = __plot_alignments(alignments, settings={
        'title': 'Attention alignment history'
    })
    plt.show()
    fig.savefig("data/blizzard/nancy/alignments.pdf", bbox_inches='tight')

    # ==============================================================================================
    # Plot attention alignment progress in a grid.
    # ==============================================================================================
    steps = []
    alignments = []

    for path in alignment_progress:
        step = path.split('-')[1].split('.')[0]
        alignment = load_attention_alignments(os.path.join(alignments_base_path, path))

        steps.append(step)
        alignments.append(alignment)

    fig = __plot_alignment_progress(steps, alignments)
    plt.show()
    fig.savefig("data/blizzard/nancy/alignments_collage.pdf", bbox_inches='tight')

    exit()

    # ==============================================================================================
    # Blizzard Nancy Train Loss (total)
    # ==============================================================================================
    fig = __plot_scalar_data([train_loss_loss['step']],
                             [train_loss_loss['value']],
                             settings={
                                 'color': [
                                     "#F8CECC",  # red
                                 ],
                                 'color_smooth': [
                                     "#B85450",  # dark red
                                 ],
                                 'title': 'Blizzard Nancy Train Loss',
                                 'xlabel': 'Steps',
                                 'ylabel': 'Loss',
                                 'ylim': (4e-2, 1.5e-1),
                                 'yticks': [4e-2, 7e-2, 8e-2, 1e-1, 1.5e-1],
                                 'yaxis_formatter': ticker.FuncFormatter(
                                     lambda y, pos: '{:.1e}'.format(y)
                                 ),
                                 'xaxis_formatter': ticker.FuncFormatter(
                                     lambda x, pos: '{:.0f}k'.format(x / 1000.0)
                                 ),
                                 'labels': ['loss', 'avg. loss']
                             })
    plt.show()
    fig.savefig("data/blizzard/nancy/loss_loss.pdf", bbox_inches='tight')

    # ==============================================================================================
    # Blizzard Nancy Train Loss (decoder)
    # ==============================================================================================
    fig = __plot_scalar_data([train_loss_decoder['step']],
                             [train_loss_decoder['value']],
                             settings={
                                 'color': [
                                     "#F8CECC",  # red
                                 ],
                                 'color_smooth': [
                                     "#B85450",  # dark red
                                 ],
                                 'title': 'Blizzard Nancy Train Loss (decoder)',
                                 'xlabel': 'Steps',
                                 'ylabel': 'Loss',
                                 'ylim': (2e-2, 1e-1),
                                 'yticks': [2e-2, 3e-2, 4e-2, 1e-1],
                                 'yaxis_formatter': ticker.FuncFormatter(
                                     lambda y, pos: '{:.1e}'.format(y)
                                 ),
                                 'xaxis_formatter': ticker.FuncFormatter(
                                     lambda x, pos: '{:.0f}k'.format(x / 1000.0)
                                 ),
                                 'labels': ['loss', 'avg. loss']
                             })
    plt.show()
    fig.savefig("data/blizzard/nancy/loss_loss_decoder.pdf", bbox_inches='tight')

    # ==============================================================================================
    # Blizzard Nancy Train Loss (post-processing)
    # ==============================================================================================
    fig = __plot_scalar_data([train_loss_post['step']],
                             [train_loss_post['value']],
                             settings={
                                 'color': [
                                     "#F8CECC",  # red
                                 ],
                                 'color_smooth': [
                                     "#B85450",  # dark red
                                 ],
                                 'title': 'Blizzard Nancy Train Loss (post-processing)',
                                 'xlabel': 'Steps',
                                 'ylabel': 'Loss',
                                 'ylim': (2e-2, 6e-2),
                                 'yticks': [2e-2, 3e-2, 4e-2, 6e-2],
                                 'yaxis_formatter': ticker.FuncFormatter(
                                     lambda y, pos: '{:.1e}'.format(y)
                                 ),
                                 'xaxis_formatter': ticker.FuncFormatter(
                                     lambda x, pos: '{:.0f}k'.format(x / 1000.0)
                                 ),
                                 'labels': ['loss', 'avg. loss']
                             })
    plt.show()
    fig.savefig("data/blizzard/nancy/loss_loss_post_processing.pdf", bbox_inches='tight')

    # ==============================================================================================
    # Blizzard Nancy Train and Evaluate Loss (total)
    # ==============================================================================================
    fig = __plot_scalar_data([
        train_loss_loss['step'], eval_loss_loss['step']
    ], [
        train_loss_loss['value'], eval_loss_loss['value']
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
            'title': 'Blizzard Nancy Train and Evaluate Loss (total)',
            'xlabel': 'Steps',
            'ylabel': 'Loss',
            'ylim': (4e-2, 8e-1),
            'yticks': [4e-2, 7e-2, 1e-1, 2e-1, 8e-1],
            'yaxis_formatter': ticker.FuncFormatter(
                lambda y, pos: '{:.0e}'.format(y)
            ),
            'xaxis_formatter': ticker.FuncFormatter(
                lambda x, pos: '{:.0f}k'.format(x / 1000.0)
            ),
            'labels': [
                'train loss',
                'avg. train loss',
                'evaluate loss',
                'avg. evaluate loss']
        })
    plt.show()
    fig.savefig("data/blizzard/nancy/loss_loss_train_eval.pdf", bbox_inches='tight')
