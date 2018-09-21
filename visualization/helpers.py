import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

def moving_avg(values, win_len, mode='same'):
    """
    Calculate the moving average over a data series.

    Arguments:
        values (np.ndarray):
            Data to calculate the moving average over.

        win_len (int):
            Size of the window used to build the average.

        mode (str):
            Padding mode. Can the following modes are supported {'full', 'valid', 'same'}.
    Returns:

    """
    weights = np.repeat(1.0, win_len) / win_len
    return np.convolve(values, weights, mode)


def plot_scalar_data(steps, values, skip=None, settings=None):
    # Default placeholder settings.
    _display = {
        'figsize': ((1.5 * 14.0 / 2.54) / 1.0, (7.7 / 2.54)),
        'color': ["#E1D5E7"],
        'color_smooth': ["#9673A6"],
        'title': None,
        'xlabel': 'Steps',
        'ylabel': 'unknown',
        'ylim': None,
        'yticks': None,
        'xaxis_formatter': None,
        'yaxis_formatter': None,
        'labels': ['unknown_{}'.format(i + 1) for i in range(len(values))],
        'ylog': True,
        'raw': True,
        'smooth': True,
        'win_len': 12
    }

    # Update the default settings with the caller settings.
    if settings is not None:
        _display.update(settings)

    fig = plt.figure(figsize=_display['figsize'], dpi=100)
    axes = [plt.gca()] * len(steps)

    # Window size used for smoothing.
    win_len = _display['win_len']

    # Reminder for the labels to show in the legend.
    labels = []

    # make sure the steps and values are numpy arrays.
    steps = np.array(steps)
    values = np.array(values)

    if skip is not None:
        # Skip the requested number of values and steps.
        steps = steps[:, skip:]
        values = values[:, skip:]

    # Iterate the individual plots.
    for i, (subplot_steps, subplot_values) in enumerate(zip(steps, values)):
        if _display['raw']:
            # Query the current plot label for the the i'th subplot.
            _label = _display['labels'][i]
            labels.append(_label)

            # Plot the i'th subplots raw values.
            if _display['ylog']:
                axes[i].semilogy(subplot_steps, subplot_values,
                             color=_display['color'][i],
                             label=_label,
                             zorder=1)
            else:
                axes[i].plot(subplot_steps, subplot_values,
                             color=_display['color'][i],
                             label=_label)

        if _display['smooth']:
            # Query the current plot label for the the i'th subplot.
            _label = 'avg. {}'.format(_display['labels'][i])
            labels.append(_label)

            # Smooth the i'th subplots values using a moving avg.
            values_smooth = moving_avg(subplot_values, win_len, 'same')

            # Do not discard data. Note that boundary effects will be visible.
            # steps_smoothed = subplot_steps
            # values_smooth = values_smooth

            # Discard values at the beginning and end where the window does not fit.
            steps_smoothed = subplot_steps[win_len - 1:-(win_len - 1)]
            values_smooth = values_smooth[win_len - 1:-(win_len - 1)]

            # Plot the i'th subplots smoothed values.
            if _display['ylog']:
                axes[i].semilogy(steps_smoothed,
                             values_smooth,
                             color=_display['color_smooth'][i],
                             label=_label,
                             zorder=2)
            else:
                axes[i].plot(steps_smoothed,
                         values_smooth,
                         color=_display['color_smooth'][i],
                         label=_label)

    axes = plt.gca()
    axes.legend(labels=labels)
    axes.grid(True, which='both', linestyle='dashed')

    axes.set_xlabel(_display['xlabel'])
    axes.set_ylabel(_display['ylabel'])

    axes.set_xlim((min(map(np.min, steps)), max(map(np.max, steps))))

    if _display['ylim'] is not None:
        axes.set_ylim(_display['ylim'])

    if _display['yticks'] is not None:
        axes.set_yticks(_display['yticks'])

    if _display['title'] is not None:
        plt.title(_display['title'])

    if _display['yaxis_formatter'] is not None:
        axes.yaxis.set_major_formatter(_display['yaxis_formatter'])
        axes.yaxis.set_minor_formatter(ticker.NullFormatter())

    if _display['xaxis_formatter'] is not None:
        axes.xaxis.set_major_formatter(_display['xaxis_formatter'])

    # Render grid above the lines.
    axes.set_axisbelow(False)

    return fig