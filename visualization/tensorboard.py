import json
import numpy as np


def load_json(path):
    """
    Load run data from a JSON file.

    Arguments:
        path (str):
            Path to the JSON file to load.

    Returns:
        data (list):
            Data loaded from the file.
    """
    with open(path, 'r') as f:
        _data = json.load(f)

    return _data


def load_scalar(path):
    """
    Load scalars exported from tensorboard from an JSON file.

    Arguments:
        path (str):
            Path to a JSON file containing scalar values exported from tensorboard.

    Returns:
        data (dict):
            Dictionary containing the parsed scalar data columns.
            The data.keys is composed of 'time', 'step' and 'value'.
    """
    _data = load_json(path)
    _data = np.array(_data)

    return {
        'time':  _data[:, [0]].flatten(),
        'step':  _data[:, [1]].flatten(),
        'value': _data[:, [2]].flatten()
    }


def load_attention_alignments(path):
    """
    Load attention alignments exported from an model.

    Arguments:
        path (str):
            Path to a .npz file containing alignments.

    Returns:
        alignments (np.ndarray):
            Copy of the attention alignments.
            The alignment history was already transposed like following:
            `alignments = tf.transpose(self.alignment_history, [1, 2, 0])`
    """
    data = np.load(path)
    return data['alignments']
