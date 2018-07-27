import matplotlib.pyplot as plt

from visualization.helpers import moving_avg
from visualization.tensorboard import load_scalar


def __plot_scalar_data(steps, values, title, ylabel, win_len=12):
    fig = plt.figure(figsize=((1.5 * 14.0 / 2.54) / 1.0, 7.7 / 2.54), dpi=100)

    # Plot the raw values.
    plt.semilogy(steps, values, color="#E1D5E7")

    # Smooth the values using a moving avg.
    values_smooth = moving_avg(values, win_len, 'same')
    # Discard values at the beginning and end where the window does not fit.
    steps_smoothed = steps[win_len - 1:-(win_len - 1)]
    values_smooth = values_smooth[win_len - 1:-(win_len - 1)]

    # Plot the smoothed values.
    plt.semilogy(steps_smoothed, values_smooth, color="#9673A6")

    plt.grid(True, which='both', linestyle='dashed')
    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel(ylabel)

    # Render grid above the lines.
    plt.rcParams['axes.axisbelow'] = False

    return fig


def plot_scalar(json_path, title, ylabel):
    data = load_scalar(json_path)
    return __plot_scalar_data(data['step'], data['value'], title, ylabel)


if __name__ == '__main__':
    plot_scalar('/tmp/run_ljspeech_LJSpeech_train-tag-loss_loss.json',
                'Loss (total)', 'Loss')
    plt.show()

    plot_scalar('/tmp/run_ljspeech_LJSpeech_train-tag-loss_loss_decoder.json',
                'Loss (decoder)',
                'Loss')
    plt.show()

    plot_scalar('/tmp/run_ljspeech_LJSpeech_train-tag-loss_loss_post_processing.json',
                'Loss (post-processing)', 'Loss')
    plt.show()
