import tensorflow as tf

from datasets.cmu_slt import CMUDatasetHelper
from datasets.lj_speech import LJSpeechDatasetHelper
from datasets.pavoque import PAVOQUEDatasetHelper
from datasets.blizzard_nancy import BlizzardNancyDatasetHelper

# Default hyper-parameters:
dataset_params = tf.contrib.training.HParams(
    # Folder containing the dataset.
    dataset_folder='/tmp/datasets_tmpfs/blizzard_nancy',

    # Dataset load helper.
    dataset_loader=BlizzardNancyDatasetHelper,

    # Vocabulary definition.
    # This definition has to include the padding and end of sequence tokens.
    # Both the padding and eos token should not be changed since they are used internally.
    vocabulary_dict={
        'pad': 0,  # padding
        'eos': 1,  # end of sequence
        # --------------------------
        'a': 2, 'c': 3, 't': 4, 'i': 5, 'n': 6, 'g': 7, ' ': 8, 'o': 9, 'u': 10, 'f': 11, 'p': 12,
        ',': 13, 'b': 14, 'e': 15, 'r': 16, 's': 17, 'd': 18, 'l': 19, 'v': 20, 'm': 21, 'w': 22,
        'h': 23, 'y': 24, 'k': 25, '-': 26, "'": 27, 'q': 28, '?': 29, 'j': 30, ':': 31, ';': 32,
        'x': 33, '!': 34, 'z': 35
        # --------------------------
    },

    # Number of unique characters in the vocabulary.
    vocabulary_size=36,
)
