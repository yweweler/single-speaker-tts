import tensorflow as tf

from datasets.blizzard_nancy import BlizzardNancyDatasetHelper
from datasets.cmu_slt import CMUDatasetHelper
from datasets.lj_speech import LJSpeechDatasetHelper
from datasets.pavoque import PAVOQUEDatasetHelper

# Default hyper-parameters:
dataset_params = tf.contrib.training.HParams(
    # Folder containing the dataset.
    dataset_folder='/thesis/datasets/cmu_us_slt_arctic',

    # Dataset load helper.
    dataset_loader=CMUDatasetHelper,

    # Vocabulary definition.
    # This definition has to include the padding and end of sequence tokens.
    # Both the padding and eos token should not be changed since they are used internally.
    vocabulary_dict={
        'pad': 0,  # padding
        'eos': 1,  # end of sequence
        # --------------------------
        'a': 2, 'u': 3, 't': 4, 'h': 5, 'o': 6, 'r': 7, ' ': 8, 'f': 9, 'e': 10, 'd': 11, 'n': 12,
        'g': 13, 'i': 14, 'l': 15, ',': 16, 'p': 17, 's': 18, 'c': 19, 'm': 20, 'z': 21, 'w': 22,
        'v': 23, 'k': 24, 'b': 25, "'": 26, 'y': 27, 'j': 28, 'q': 29, 'x': 30, '-': 31, ';': 32
        # --------------------------
    },

    # Number of unique characters in the vocabulary.
    vocabulary_size=33,
)
