import numpy as np

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