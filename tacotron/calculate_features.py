"""
Pre-compute features for a dataset and store them on disk.
"""

import os
from itertools import chain, repeat
from multiprocessing.pool import ThreadPool

import numpy as np

from datasets.dataset import Dataset
from datasets.utils.processing import py_calculate_spectrogram, load_audio
from tacotron.params.dataset import dataset_params
from tacotron.params.model import model_params


def __pre_compute(args):
    _path, normalization_params = args

    # Load the audio file from disk.
    audio, _ = load_audio(_path)

    # Calculate normalized and processed spectrogram features.
    mel_mag_db, linear_mag_db = py_calculate_spectrogram(
        normalization_params['mel_mag_ref_db'],
        normalization_params['mel_mag_max_db'],
        normalization_params['linear_ref_db'],
        normalization_params['linear_mag_max_db'],
        model_params.win_len,
        model_params.win_hop,
        model_params.sampling_rate,
        model_params.n_fft,
        model_params.n_mels,
        model_params.mel_fmin,
        model_params.mel_fmax,
        model_params.reduction,
        audio
    )

    # Extract the file path and file name without the extension.
    file_path = os.path.splitext(_path)[0]

    # Create the target file path.
    out_path = '{}.npz'.format(file_path)
    print('Writing: "{}"'.format(out_path))

    # Save the audio file as a numpy .npz file.
    np.savez(out_path, mel_mag_db=mel_mag_db, linear_mag_db=linear_mag_db)


def pre_compute_features(dataset_loader, n_threads=1):
    """
    Loads all audio files from the dataset, computes features and saves these pre-computed
    features as numpy .npz files to disk.

    For example: The features of an audio file <path>/<filename>.wav are saved next to the
    audio file in <path>/<filename>.npz.

    Arguments:
        dataset_loader (datasets.dataset.Dataset):
            Dataset loading helper to load the dataset with.

        n_threads (int):
            Number of threads to use for parallel processing.
            Default is `1`.
    """
    # Collect the train and eval audio file paths.
    paths = chain(
        (row['audio_path'] for row in dataset_loader.get_train_listing_generator()),
        (row['audio_path'] for row in dataset_loader.get_eval_listing_generator())
    )

    normalization_params = dataset_loader.get_normalization()
    call_arguments = zip(paths, repeat(normalization_params))

    # Pre-compute and save features for all audio files in the listing.
    pool = ThreadPool(n_threads)
    pool.map(__pre_compute, call_arguments)
    pool.close()
    pool.join()


if __name__ == '__main__':
    # Load the dataset from a dataset definition file.
    dataset = Dataset(dataset_params.dataset_file)
    dataset.load()

    # Load the corresponding train and eval listing files.
    dataset.load_listings()

    # Pre-calculate features and store them on disk.
    pre_compute_features(dataset, n_threads=4)
