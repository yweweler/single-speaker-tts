import json
import os

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


def load_scalar(path, steps=None):
    """
    Load scalars exported from tensorboard from an JSON file.

    Arguments:
        path (str):
            Path to a JSON file containing scalar values exported from tensorboard.

        steps (int):
            Number of steps till which to load data rows from the file.
            If None, all the entire file is loaded.
            If more steps are requested than are available, only the available steps are returned.

    Returns:
        data (dict):
            Dictionary containing the parsed scalar data columns.
            The data.keys is composed of 'time', 'step' and 'value'.
    """
    _data = load_json(path)
    _data = np.array(_data)

    # Collect the timestamps.
    _time_array = _data[:, [0]].flatten()

    # Collect the step counts.
    _step_array = _data[:, [1]].flatten()

    # Collect the loss values.
    _value_array = _data[:, [2]].flatten()

    max_index = len(_time_array)
    if steps is not None:
        # Determine the highest index to load.
        # Keep in mind that the stored values also include values for 0 train/eval iterations.
        for i, current_step in enumerate(_step_array):
            # print(i, current_step)
            if steps <= current_step:
                max_index = i
                break

    return {
        'time': _time_array[:max_index + 1],
        'step': _step_array[:max_index + 1],
        'value': _value_array[:max_index + 1]
    }


def load_run(folder, steps=None):
    """

    Arguments:
        folder (str):
            Path to the folder containing the run files.

        steps (int):
            Number of steps till which to load data rows from the file.
            See: `visualization.helpers.load_scalar` for more details.

    Returns:
        data (dict):
            Dictionary containing the parsed scalar data columns for train and eval runs.
            The data.keys is composed of 'eval-{total, decoder, post}' and 'train-{total,
            decoder, post}'.
    """
    _files = {
        'eval-total': 'evaluate-loss_loss.json',
        'eval-decoder': 'evaluate-loss_decoder.json',
        'eval-post': 'evaluate-loss_post_processing.json',
        'train-total': 'train-loss_loss.json',
        'train-decoder': 'train-loss_decoder.json',
        'train-post': 'train-loss_post_processing.json',
    }

    _data = dict()
    for run in _files:
        _run_file = os.path.join(folder, _files[run])
        _run_data = load_scalar(path=_run_file, steps=steps)
        _data[run] = _run_data

    return _data


def merge_runs_by_tag(runs, tags):
    """
    Collect the (step, value) tuples corresponding to individual tags for all runs.

    Therefore the result might look like this:
    <tagA>
      + step:
        - <run-1-steps>
        - <run-2-steps>
      + value:
        - <run-1-values>
        - <run-2-values>
      ...

    Arguments:
        runs (dict):
            Collection of data from all runs.
            Usually the output of `visualization.helpers.load_run`

        tags (list):
            List of the tags to merge.

    Returns:
        data (dict):
            Dictionary containing the merged scalar data of all runs.
            The data.keys is composed of `tags`.
    """
    _merged_runs = dict()

    # Merge the steps and values for each tag over all runs.
    for tag in tags:
        _run_values = [runs[run][tag]['value'] for run in runs]
        _run_steps = [runs[run][tag]['step'] for run in runs]
        _merged_runs[tag] = {
            'step': _run_steps,
            'value': _run_values
        }

    return _merged_runs


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


