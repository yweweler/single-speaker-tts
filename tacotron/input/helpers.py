import numpy as np
import tensorflow as tf
from datasets.dataset_helper import DatasetHelper
from tacotron.params.dataset import dataset_params
from tacotron.params.model import model_params
from tacotron.params.training import training_params
from tensorflow.python.data.experimental.ops import grouping
import sys


def derive_bucket_boundaries(element_lengths, n_buckets):
    # Get the total number of samples in the dataset.
    n_samples = len(element_lengths)

    # Sort sequence lengths in order to slice them into buckets that contain sequences of roughly
    # equal length.
    sorted_sentence_lengths = np.sort(element_lengths)

    if n_samples < n_buckets:
        raise AssertionError('The number of entries loaded is smaller than the number of '
                             'buckets to be created. Automatic calculation of the bucket '
                             'boundaries is not possible.')

    # Slice the sorted lengths into equidistant sections and use the first element of a slice as
    # the bucket boundary.
    bucket_step = n_samples // n_buckets
    bucket_boundaries = sorted_sentence_lengths[::bucket_step]

    # Throw away the first and last bucket boundaries since the bucketing algorithm automatically
    # adds two surrounding ones.
    bucket_boundaries = bucket_boundaries[1:-1].tolist()

    # Remove duplicate boundaries from the list.
    bucket_boundaries = sorted(list(set(bucket_boundaries)))

    print('bucket_boundaries', bucket_boundaries)
    print('n_buckets: {} + 2'.format(len(bucket_boundaries)))

    return bucket_boundaries
