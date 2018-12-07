import os


def inputs_from_dataset_iter(dataset_iter):
    # get inputs from the dataset iterator.
    ph_sentences, ph_sentence_lengths, ph_mel_specs, ph_lin_specs, ph_time_frames = \
        dataset_iter.get_next()

    inputs = {
        'ph_sentences': ph_sentences,
        'ph_sentence_length': ph_sentence_lengths,
        'ph_mel_specs': ph_mel_specs,
        'ph_lin_specs': ph_lin_specs,
        'ph_time_frames': ph_time_frames
    }

    return inputs


def collect_checkpoint_paths(checkpoint_dir):
    """
    Generates a list of paths to each checkpoint file found in a folder.

    Note:
        - This function assumes, that checkpoint paths were written in relative form.

    Arguments:
        checkpoint_dir (string):
            Path to the models checkpoint directory from which to collect checkpoints.

    Returns:
        paths (:obj:`list` of :obj:`string`):
            List of paths to each checkpoint file.
    """
    listing_file = os.path.join(checkpoint_dir, 'checkpoint')
    lines = []

    # Collect all lines from the checkpoint listing file.
    for line in open(listing_file, 'r'):
        line = line.strip()
        lines.append(line)

    # Discard the first line since it only points to the latest checkpoint.
    lines = lines[1:]

    # Extract the checkpoints path and global step from each line.
    # NOTE: This functions assumes, that all checkpoint paths are relative.
    # all_model_checkpoint_paths: "model.ckpt-<global-step>"

    # Remove "all_model_checkpoint_paths: " from each line.
    lines = [line.replace('all_model_checkpoint_paths: ', '') for line in lines]

    # Remove surrounding quotation marks (" .. ") from each line.
    lines = [line.replace('"', '') for line in lines]

    # Extract the global step from each line.
    # steps = [int(line.split('-', 1)[-1]) for line in lines]

    # Build absolute paths to each checkpoint file.
    paths = [os.path.join(checkpoint_dir, line) for line in lines]

    return paths
