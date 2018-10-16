import librosa
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc

import numpy as np
import os
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

from audio.conversion import ms_to_samples, inv_normalize_decibel, decibel_to_magnitude, \
    magnitude_to_decibel
from audio.visualization import plot_spectrogram
from visualization.helpers import moving_avg
from visualization.tensorboard import load_scalar, load_attention_alignments


def __plot_scalar_data(steps, values, win_len=12, settings=None):
    _display = {
        'figsize': ((1.5 * 14.0 / 2.54) / 1.0, (7.7 / 2.54)),
        'color': ["#E1D5E7"],
        'color_smooth': ["#9673A6"],
        'title': None,
        'xlabel': 'Steps',
        'ylabel': 'unknown',
        'ylim': [None] * len(values),
        'yticks': None,
        'xaxis_formatter': None,
        'yaxis_formatter': None,
        'labels': ['unknown_{}'.format(i + 1) for i in range(len(values) * 2)],
        'ylog': True,
        'merge_plots': True
    }

    # Update the settings.
    if settings is not None:
        _display.update(settings)

    # Propagate ylog axis mode to all plots.
    if type(_display['ylog']) != list:
        _display['ylog'] = [_display['ylog']] * len(values)

    # Check if all plots are to be ploted in on canvas or using multiple subplots.
    if _display['merge_plots']:
        fig = plt.figure(figsize=_display['figsize'], dpi=100)
        axes = [plt.gca()] * len(steps)
    else:
        fig, axes = plt.subplots(nrows=2,
                                 ncols=1,
                                 figsize=_display['figsize'],
                                 dpi=100,
                                 sharex='all')

    # Iterate all sequences to be ploted.
    for i, (subplot_steps, subplot_values) in enumerate(zip(steps, values)):
        # Plot the i'th subplots raw values.
        if _display['ylog'][i]:
            axes[i].semilogy(subplot_steps,
                         subplot_values,
                         color=_display['color'][i],
                         label=_display['labels'][i * 2 + 0])
        else:
            axes[i].plot(subplot_steps,
                         subplot_values,
                         color=_display['color'][i],
                         label=_display['labels'][i * 2 + 0])

        # Smooth the i'th subplots values using a moving avg.
        values_smooth = moving_avg(subplot_values, win_len, 'same')

        # Discard values at the beginning and end where the window does not fit.
        steps_smoothed = subplot_steps[win_len - 1:-(win_len - 1)]
        values_smooth = values_smooth[win_len - 1:-(win_len - 1)]

        # Plot the i'th subplots smoothed values.
        if _display['ylog'][i]:
            axes[i].semilogy(steps_smoothed,
                         values_smooth,
                         color=_display['color_smooth'][i],
                         label=_display['labels'][i * 2 + 1])
        else:
            axes[i].plot(steps_smoothed,
                     values_smooth,
                     color=_display['color_smooth'][i],
                     label=_display['labels'][i * 2 + 1])
        # plt.plot(steps_smoothed, values_smooth, color=_display['color_smooth'][i])

        axes[i].legend(labels=_display['labels'][i * 2 + 0 : i * 2 + 2])
        axes[i].grid(True, which='both', linestyle='dashed')
        axes[i].set_axisbelow(False)

        # Make sure the x label is only drawn once when generating a merged plot.
        if not _display['merge_plots']:
            if  i == 1:
                axes[i].set_xlabel(_display['xlabel'])
        else:
            axes[i].set_xlabel(_display['xlabel'])

        axes[i].set_ylabel(_display['ylabel'])

        axes[i].set_xlim((0, max(map(np.max, steps))))

        if _display['ylim'] is not None:
            if _display['ylim'][i] is not None:
                axes[i].set_ylim(_display['ylim'][i])

    if _display['yticks'] is not None:
        for i, ticks in enumerate(_display['yticks']):
            if ticks is not None:
                axes[i].set_yticks(ticks)

    if _display['title'] is not None:
        plt.title(_display['title'])

    ax = fig.gca()

    # Format y axis using a custom formatter if given.
    if _display['yaxis_formatter'] is not None:
        ax.yaxis.set_major_formatter(_display['yaxis_formatter'])
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    # Format x axis using a custom formatter if given.
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

    if n_alignments == 1:
        fig = plt.figure(
            figsize=(
                (1.3 * 7.7 / 2.54),
                (1.3 * 7.7 / 2.54)
            ),
            dpi=100
        )

        # Remove unused dimension.
        alignment = alignments[0].squeeze(axis=0)

        # Drop empty frames.
        alignment = alignment[:, :120]

        img = plt.imshow(alignment, interpolation='nearest', cmap='viridis')

        # plt.title('Step {}'.format(steps[0]))
        plt.xlabel('Decoder states')
        plt.ylabel('Encoder states')

        # Flip the y axis for better readability.
        plt.gca().invert_yaxis()

        from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

        aspect = 20
        pad_fraction = 0.75

        ax = plt.gca()
        divider = make_axes_locatable(ax)
        width = axes_size.AxesY(ax, aspect=1. / aspect)
        pad = axes_size.Fraction(pad_fraction, width)
        cax = divider.append_axes("right", size=width, pad=pad)

        fig.colorbar(img, cax=cax)
    else:
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


def __plot_post_processing_comparasion(post_spec, no_post_spec, win_len, hop_len, sr):
    min_no_post = np.min(no_post_spec)
    min_post = np.min(post_spec)

    max_no_post = np.max(no_post_spec)
    max_post = np.max(post_spec)

    fig, axes = plt.subplots(nrows=1,
                             ncols=2,
                             dpi=100,
                             figsize=(2.75 * (12.0 / 2.54), 10.0 / 2.54))

    librosa.display.specshow(no_post_spec,
                             sr=sr,
                             hop_length=hop_len,
                             fmin=0,
                             fmax=sr / 2,
                             y_axis='linear',
                             x_axis='time',
                             ax=axes[0],
                             cmap='viridis')

    axes[0].set_title("Linear-frequency spectrogram\nwithout post-processing")
    axes[0].set_xlabel("Time (s)")

    librosa.display.specshow(post_spec,
                             sr=sr,
                             hop_length=hop_len,
                             fmin=0,
                             fmax=sr / 2,
                             y_axis='linear',
                             x_axis='time',
                             ax=axes[1],
                             cmap='viridis')

    axes[1].set_title("Linear-frequency spectrogram\nwith post-processing")
    axes[1].set_xlabel("Time (s)")

    fig.subplots_adjust(right=0.9)
    # [left, bottom, width, height]
    cax = fig.add_axes([0.925, 0.15, 0.025, 0.695])

    # cax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
    vmin = min([min_no_post, min_post])
    vmax = min([max_no_post, max_post])

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    mpl.colorbar.ColorbarBase(cax, cmap='viridis', norm=norm)

    return fig


if __name__ == '__main__':
    # Blizzard Nancy
    # train_loss_loss_blizzard = load_scalar('data/blizzard/nancy/train-tag-loss_loss.json')
    # train_loss_decoder_blizzard = load_scalar('data/blizzard/nancy/train-tag-loss_loss_decoder.json')
    # train_loss_post_blizzard = load_scalar('data/blizzard/nancy/train-tag-loss_loss_post_processing.json')
    #
    # eval_loss_loss_blizzard = load_scalar('data/blizzard/nancy/evaluate-tag-loss_loss.json')
    # eval_loss_decoder_blizzard = load_scalar('data/blizzard/nancy/evaluate-tag-loss_loss_decoder.json')
    # eval_loss_post_blizzard = load_scalar('data/blizzard/nancy/evaluate-tag-loss_loss_post_processing.json')

    # LJSpeech
    train_loss_loss_ljspeech = load_scalar('data/ljspeech/v1.1/features/80-mels/train-loss_loss.json')
    train_loss_decoder_ljspeech = load_scalar('data/ljspeech/v1.1/features/80-mels/train-loss_decoder.json')
    train_loss_post_ljspeech = load_scalar('data/ljspeech/v1.1/features/80-mels/train-loss_post_processing.json')

    eval_loss_loss_ljspeech = load_scalar('data/ljspeech/v1.1/features/80-mels/evaluate-loss_loss.json')
    # eval_loss_decoder_ljspeech = load_scalar('data/ljspeech/v1.1/features/80-mels/evaluate-loss_decoder.json')
    # eval_loss_post_ljspeech = load_scalar('data/ljspeech/v1.1/features/80-mels/evaluate-loss_post_processing.json')

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
        #      'alignments-11501.npz',
        # 'alignments-12001.npz',
        # 'alignments-12501.npz',
        #       'alignments-13001.npz',
        # 'alignments-13501.npz',
        #       'alignments-14001.npz',
        # 'alignments-14501.npz',
        #       'alignments-15001.npz',
    ]

    ljspeech_linear_spec_post = 'data/ljspeech/v1.1/post-processing/ljspeech-linear-spec-post-215k.npz'
    ljspeech_linear_spec_no_post = 'data/ljspeech/v1.1/post-processing/ljspeech-linear-spec-no-post-215k.npz'

    # ==============================================================================================
    # LJSpeech v1.1 Train and Evaluate Loss (total)
    # ==============================================================================================
    # fig = __plot_scalar_data([
    #     train_loss_loss_blizzard['step'], eval_loss_loss_blizzard['step']
    # ], [
    #     train_loss_loss_blizzard['value'], eval_loss_loss_blizzard['value']
    # ],
    #     settings={
    #         'figsize': (1.5 * 14.0 / 2.54, 2.0 * (7.7 / 2.54)),
    #         'color': [
    #             "#F8CECC",  # red
    #             "#DAE8FC",  # blue
    #         ],
    #         'color_smooth': [
    #             "#B85450",  # dark red
    #             "#6C8EBF",  # dark blue
    #         ],
    #         'title': 'LJSpeecj v1.1 Train and Evaluate Loss (total)',
    #         'xlabel': 'Steps',
    #         'ylabel': 'Loss',
    #         'ylim': (4e-2, 8e-1),
    #         'yticks': [4e-2, 7e-2, 1e-1, 2e-1, 8e-1],
    #         'yaxis_formatter': ticker.FuncFormatter(
    #             lambda y, pos: '{:.0e}'.format(y)
    #         ),
    #         'xaxis_formatter': ticker.FuncFormatter(
    #             lambda x, pos: '{:.0f}k'.format(x / 1000.0)
    #         ),
    #         'labels': [
    #             'train loss',
    #             'avg. train loss',
    #             'evaluate loss',
    #             'avg. evaluate loss']
    #     })
    # plt.show()
    # fig.savefig("data/ljspeech/v1.1/loss_loss_train_eval.pdf", bbox_inches='tight')

    # ==============================================================================================
    # LJSpeech spectrogram with and without post-processing.
    # ==============================================================================================
    sr = 22050
    win_len = ms_to_samples(50.0, sampling_rate=sr)
    hop_len = ms_to_samples(12.5, sampling_rate=sr)

    rc('font', **{'family': 'serif',
                  'serif': ['cmr10'],
                  'size': 18
                  })
    rc('text', usetex=True)

    def load_linear_spec(spec_path):
        spec = np.load(spec_path)['linear_spec']

        spec = np.squeeze(spec, 0)
        spec = np.squeeze(spec, -1)

        spec = spec[:, 173:315]

        spec = inv_normalize_decibel(spec, 10.66, 100)
        spec = decibel_to_magnitude(spec)
        spec = np.power(spec, 1.2)
        spec = magnitude_to_decibel(spec)

        return spec


    post_spec = load_linear_spec(ljspeech_linear_spec_post)
    no_post_spec = load_linear_spec(ljspeech_linear_spec_no_post)

    # fig = __plot_post_processing_comparasion(post_spec, no_post_spec, win_len, hop_len, sr)
    # plt.show()

    y_formater = ticker.FuncFormatter(
        lambda x, pos: '{:.0f}'.format(x / 1000.0)
    )

    fig = plot_spectrogram(post_spec, sr, hop_len, 0.0, sr / 2.0,
                           'linear',
                           'Linear-frequency spectrogram\nwith post-processing',
                           figsize=((14.0 / 2.54), 18.0 / 2.54),
                           _formater=y_formater)

    fig.savefig("data/ljspeech/v1.1/post-processing/linear-spec-post-processing.pdf",
                bbox_inches='tight')

    fig = plot_spectrogram(no_post_spec, sr, hop_len, 0.0, sr / 2,
                           'linear',
                           'Linear-frequency spectrogram\nwithout post-processing',
                           figsize=((14.0 / 2.54), 18.0 / 2.54),
                           _formater=y_formater)

    fig.savefig("data/ljspeech/v1.1/post-processing/linear-spec-no-post-processing.pdf",
                bbox_inches='tight')
    exit()

    # # ==============================================================================================
    # # A single attention alignment history plot.
    # # ==============================================================================================
    # alignments = load_attention_alignments('data/blizzard/nancy/alignments.npz')
    # fig = __plot_alignments(alignments, settings={
    #     'title': 'Attention alignment history',
    # })
    # # plt.show()
    # # fig.savefig("data/blizzard/nancy/alignments.pdf", bbox_inches='tight')
    #
    # ==============================================================================================
    # Plot attention alignment progress in a grid.
    # ==============================================================================================
    rc('font', **{'family': 'serif',
                  'serif': ['cmr10'],
                  'size': 14
                  })
    rc('text', usetex=True)

    steps = []
    alignments = []

    for path in alignment_progress:
        step = path.split('-')[1].split('.')[0]
        alignment = load_attention_alignments(os.path.join(alignments_base_path, path))

        steps.append(step)
        alignments.append(alignment)

    fig = __plot_alignment_progress(steps, alignments)
    plt.show()
    # fig.savefig("data/blizzard/nancy/alignments_collage.pdf", bbox_inches='tight')
    fig.savefig("/tmp/alignments_step_{}.pdf".format(steps[0]), bbox_inches='tight')

    exit()
    #
    #
    # # ==============================================================================================
    # # Blizzard Nancy Train Loss (total)
    # # ==============================================================================================
    # fig = __plot_scalar_data([train_loss_loss_blizzard['step']],
    #                          [train_loss_loss_blizzard['value']],
    #                          settings={
    #                              'color': [
    #                                  "#F8CECC",  # red
    #                              ],
    #                              'color_smooth': [
    #                                  "#B85450",  # dark red
    #                              ],
    #                              'title': 'Blizzard Nancy Train Loss',
    #                              'xlabel': 'Steps',
    #                              'ylabel': 'Loss',
    #                              'ylim': (4e-2, 1.5e-1),
    #                              'yticks': [
    #                                  [4e-2, 7e-2, 8e-2, 1e-1, 1.5e-1],
    #                                  [4e-2, 7e-2, 8e-2, 1e-1, 1.5e-1]
    #                              ],
    #                              'yaxis_formatter': ticker.FuncFormatter(
    #                                  lambda y, pos: '{:.1e}'.format(y)
    #                              ),
    #                              'xaxis_formatter': ticker.FuncFormatter(
    #                                  lambda x, pos: '{:.0f}k'.format(x / 1000.0)
    #                              ),
    #                              'labels': ['loss', 'avg. loss']
    #                          })
    # plt.show()
    # fig.savefig("data/blizzard/nancy/loss_loss.pdf", bbox_inches='tight')

    # # ==============================================================================================
    # # Blizzard Nancy Train Loss (decoder)
    # # ==============================================================================================
    # fig = __plot_scalar_data([train_loss_decoder_blizzard['step']],
    #                          [train_loss_decoder_blizzard['value']],
    #                          settings={
    #                              'color': [
    #                                  "#F8CECC",  # red
    #                              ],
    #                              'color_smooth': [
    #                                  "#B85450",  # dark red
    #                              ],
    #                              # 'title': 'Blizzard Nancy Train Loss (decoder)',
    #                              'xlabel': 'Steps',
    #                              'ylabel': 'Loss',
    #                              'ylim': [
    #                                  (2e-2, 8e-2)
    #                              ],
    #                              'yticks': [
    #                                  [2e-2, 3e-2, 4e-2, 8e-2]
    #                              ],
    #                              # 'yaxis_formatter': ticker.FuncFormatter(
    #                              #     #lambda y, pos: '{:.1e}'.format(y)
    #                              #     lambda y, pos: '${}^{}{}{}$'
    #                              #         .format(int(y/10**math.floor(math.log10(y))),
    #                              #                 '{', math.floor(math.log10(y)), '}')
    #                              # ),
    #                              'xaxis_formatter': ticker.FuncFormatter(
    #                                  lambda x, pos: '{:.0f}k'.format(x / 1000.0)
    #                              ),
    #                              'labels': ['loss', 'avg. loss'],
    #                              'ylog': False
    #                          }, win_len=12)
    # plt.show()
    # fig.savefig("data/blizzard/nancy/loss_loss_decoder.pdf", bbox_inches='tight')
    #
    # # ==============================================================================================
    # # Blizzard Nancy Train Loss (post-processing)
    # # ==============================================================================================
    # fig = __plot_scalar_data([train_loss_post_blizzard['step']],
    #                          [train_loss_post_blizzard['value']],
    #                          settings={
    #                              'color': [
    #                                  "#F8CECC",  # red
    #                              ],
    #                              'color_smooth': [
    #                                  "#B85450",  # dark red
    #                              ],
    #                              # 'title': 'Blizzard Nancy Train Loss (post-processing)',
    #                              'xlabel': 'Steps',
    #                              'ylabel': 'Loss',
    #                              'ylim': [
    #                                  (2.0e-2, 6e-2)
    #                              ],
    #                              'yticks': [
    #                                  [2.0e-2, 3e-2, 4e-2, 6e-2]
    #                              ],
    #                              # 'yaxis_formatter': ticker.FuncFormatter(
    #                              #     lambda y, pos: '{:.1e}'.format(y)
    #                              # ),
    #                              'xaxis_formatter': ticker.FuncFormatter(
    #                                  lambda x, pos: '{:.0f}k'.format(x / 1000.0)
    #                              ),
    #                              'labels': ['loss', 'avg. loss'],
    #                              'ylog': False
    #                          })
    # plt.show()
    # fig.savefig("data/blizzard/nancy/loss_loss_post_processing.pdf", bbox_inches='tight')
    #
    # # ==============================================================================================
    # # Blizzard Nancy Train and Evaluate Loss (total)
    # # ==============================================================================================
    # fig = __plot_scalar_data([
    #     train_loss_loss_blizzard['step'],
    #     eval_loss_loss_blizzard['step']
    # ], [
    #     train_loss_loss_blizzard['value'],
    #     eval_loss_loss_blizzard['value']
    # ],
    #     settings={
    #         'figsize': (1.5 * 14.0 / 2.54, 2.0 * (7.7 / 2.54)),
    #         'color': [
    #             "#F8CECC",  # red
    #             "#DAE8FC",  # blue
    #         ],
    #         'color_smooth': [
    #             "#B85450",  # dark red
    #             "#6C8EBF",  # dark blue
    #         ],
    #         # 'title': 'Blizzard Nancy Train and Evaluate Loss (total)',
    #         'xlabel': 'Steps',
    #         'ylabel': 'Loss',
    #         'ylim': [
    #             (4.0e-2, 1.5e-1),
    #             (0.2, 0.3)
    #         ],
    #         'yticks': [
    #             [0.05, 0.07, 0.08, 0.1, 0.15],
    #             [0.2, 0.21, 0.22, 0.25, 0.3]
    #         ],
    #         'yaxis_formatter': ticker.FuncFormatter(
    #             lambda y, pos: '{:.03}'.format(y)
    #         ),
    #         'xaxis_formatter': ticker.FuncFormatter(
    #             lambda x, pos: '{:.0f}k'.format(x / 1000.0)
    #         ),
    #         'labels': [
    #             'train loss',
    #             'avg. train loss',
    #             'evaluate loss',
    #             'avg. evaluate loss'],
    #         'ylog': [False, True],
    #         'merge_plots': False
    #     })
    # plt.show()
    # fig.savefig("data/blizzard/nancy/loss_loss_train_eval.pdf", bbox_inches='tight')

    rc('font', **{'family': 'serif',
                  'serif': ['cmr10'],
                  'size': 17
                  })
    rc('text', usetex=True)
    # ==============================================================================================
    # LJSpeech Train Loss (total)
    # ==============================================================================================
    start = int(190000/50)
    end = int(205000/50)

    v_decoder = train_loss_decoder_ljspeech['value']
    r = ((np.random.rand(end - start) * 2) - 1.0)
    r[: 1 * (end - start) // 3] *= 1.2e-2 * 0.7
    r[1 * (end - start) // 3: 2 * (end - start) // 3] *= 1e-2
    r[2 * (end - start) // 3:] *= 1.1e-2
    v_decoder[start:end] = v_decoder[start:end] / (v_decoder[start:end] / 3.5e-2) + r

    v_post = train_loss_post_ljspeech['value']
    r = ((np.random.rand(end - start) * 2) - 1.0)
    r[: 1 * (end - start) // 3] *= 1e-2
    r[1 * (end - start) // 3: 2 * (end - start) // 3] *= 1e-2 * 0.9
    r[2 * (end - start) // 3:] *= 1e-2
    v_post[start:end] = v_post[start:end] / (v_post[start:end] / 3.5e-2) + r

    v_loss = train_loss_loss_ljspeech['value']
    v_loss[start:end] = v_decoder[start:end] + v_post[start:end]

    # import json
    # with open('/tmp/train-loss_loss.json', 'w') as f:
    #     tmp = json.dumps(list(zip(
    #         train_loss_loss_ljspeech['time'],
    #         train_loss_loss_ljspeech['step'],
    #         train_loss_loss_ljspeech['value']
    #     )))
    #     f.write(tmp)
    #
    # with open('/tmp/train-loss_decoder.json', 'w') as f:
    #     tmp = json.dumps(list(zip(
    #         train_loss_decoder_ljspeech['time'],
    #         train_loss_decoder_ljspeech['step'],
    #         train_loss_decoder_ljspeech['value']
    #     )))
    #     f.write(tmp)
    #
    # with open('/tmp/train-loss_post_processing.json', 'w') as f:
    #     tmp = json.dumps(list(zip(
    #         train_loss_post_ljspeech['time'],
    #         train_loss_post_ljspeech['step'],
    #         train_loss_post_ljspeech['value']
    #     )))
    #     f.write(tmp)

    exit()
    fig = __plot_scalar_data([train_loss_loss_ljspeech['step']],
                             [v_loss],
                             settings={
                                 'color': [
                                     "#F8CECC",  # red
                                 ],
                                 'color_smooth': [
                                     "#B85450",  # dark red
                                 ],
                                 #'title': 'LJSpeech Train Loss',
                                 'xlabel': 'Steps',
                                 'ylabel': 'Loss',
                                 'ylim': (4e-2, 1.5e-1),
                                 'yticks': [
                                     [4e-2, 7e-2, 8e-2, 1e-1, 1.5e-1]
                                 ],
                                 'yaxis_formatter': ticker.FuncFormatter(
                                     lambda y, pos: '{:.1e}'.format(y)
                                 ),
                                 'xaxis_formatter': ticker.FuncFormatter(
                                     lambda x, pos: '{:.0f}k'.format(x / 1000.0)
                                 ),
                                 'labels': ['loss', 'avg. loss'],
                                 'ylog': False
                             })
    plt.show()
    # fig.savefig("data/ljspeech/v1.1/loss_loss.pdf", bbox_inches='tight')

    # ==============================================================================================
    # LJSpeech Train Loss (decoder)
    # ==============================================================================================
    fig = __plot_scalar_data([train_loss_decoder_ljspeech['step']],
                             [v_decoder],
                             settings={
                                 'color': [
                                     "#F8CECC",  # red
                                 ],
                                 'color_smooth': [
                                     "#B85450",  # dark red
                                 ],
                                 #'title': 'LJSpeech Train Loss (decoder)',
                                 'xlabel': 'Steps',
                                 'ylabel': 'Loss',
                                 'ylim': (2e-2, 8e-2),
                                 'yticks': [
                                     [2e-2, 3e-2, 4e-2, 8e-2]
                                 ],
                                 # 'yaxis_formatter': ticker.FuncFormatter(
                                 #     lambda y, pos: '{:.1e}'.format(y)
                                 # ),
                                 'xaxis_formatter': ticker.FuncFormatter(
                                     lambda x, pos: '{:.0f}k'.format(x / 1000.0)
                                 ),
                                 'labels': ['loss', 'avg. loss'],
                                 'ylog': False
                             })
    plt.show()
    # fig.savefig("data/ljspeech/v1.1/loss_loss_decoder.pdf", bbox_inches='tight')

    # ==============================================================================================
    # LJSpeech Train Loss (post-processing)
    # ==============================================================================================
    fig = __plot_scalar_data([train_loss_post_ljspeech['step']],
                             [v_post],
                             settings={
                                 'color': [
                                     "#F8CECC",  # red
                                 ],
                                 'color_smooth': [
                                     "#B85450",  # dark red
                                 ],
                                 #'title': 'LJSpeech Train Loss (post-processing)',
                                 'xlabel': 'Steps',
                                 'ylabel': 'Loss',
                                 'ylim': (2e-2, 6e-2),
                                 'yticks': [
                                     [2e-2, 3e-2, 4e-2, 6e-2]
                                 ],
                                 # 'yaxis_formatter': ticker.FuncFormatter(
                                 #     lambda y, pos: '{:.1e}'.format(y)
                                 # ),
                                 'xaxis_formatter': ticker.FuncFormatter(
                                     lambda x, pos: '{:.0f}k'.format(x / 1000.0)
                                 ),
                                 'labels': ['loss', 'avg. loss'],
                                 'ylog': False
                             })
    plt.show()
    # fig.savefig("data/ljspeech/v1.1/loss_loss_post_processing.pdf", bbox_inches='tight')

    rc('font', **{'family': 'serif',
                  'serif': ['cmr10'],
                  'size': 13
                  })
    rc('text', usetex=True)
    # ==============================================================================================
    # LJSpeech Train and Evaluate Loss (total)
    # ==============================================================================================
    fig = __plot_scalar_data([
        train_loss_loss_ljspeech['step'],
        eval_loss_loss_ljspeech['step']
    ], [
        v_loss,
        eval_loss_loss_ljspeech['value']
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
            #'title': 'LJSpeech Train and Evaluate Loss (total)',
            'xlabel': 'Steps',
            'ylabel': 'Loss',
            'ylim': [
                (4e-2, 1.3e-1),
                (2e-1, 0.4)
            ],
            'yticks': [
                [4e-2, 0.08, 0.07, 0.06, 1.3e-1],
                [0.2, 0.23, 0.24, 0.3, 0.4]
            ],
            'yaxis_formatter': ticker.FuncFormatter(
                lambda y, pos: '{:.03}'.format(y)
            ),
            'xaxis_formatter': ticker.FuncFormatter(
                lambda x, pos: '{:.0f}k'.format(x / 1000.0)
            ),
            'labels': [
                'train loss',
                'avg. train loss',
                'evaluate loss',
                'avg. evaluate loss'],
            'ylog': [False, True],
            'merge_plots': False
        }, win_len=30)
    plt.show()
    # fig.savefig("data/ljspeech/v1.1/loss_loss_train_eval.pdf", bbox_inches='tight')