import tensorflow as tf

from datasets.blizzard_nancy import BlizzardNancyDatasetHelper
from datasets.cmu_slt import CMUDatasetHelper
from datasets.lj_speech import LJSpeechDatasetHelper
from datasets.pavoque import PAVOQUEDatasetHelper

# Default hyper-parameters:
dataset_params = tf.contrib.training.HParams(
    # Folder containing the dataset.
    dataset_folder='/thesis/datasets/ljspeech',

    # Dataset load helper.
    dataset_loader=LJSpeechDatasetHelper,

    # Vocabulary definition.
    # This definition has to include the padding and end of sequence tokens.
    # Both the padding and eos token should not be changed since they are used internally.
    vocabulary_dict={
        'pad': 0,  # padding
        'eos': 1,  # end of sequence
        # --------------------------
        'p': 2, 'r': 3, 'i': 4, 'n': 5, 't': 6, 'g': 7, ' ': 8, 'h': 9, 'e': 10, 'o': 11, 'l': 12,
        'y': 13, 's': 14, 'w': 15, 'c': 16, 'a': 17, 'd': 18, 'f': 19, 'm': 20, 'x': 21, 'b': 22,
        'v': 23, 'u': 24, 'k': 25, 'j': 26, 'z': 27, 'q': 28, ',': 29, '"': 30, '-': 31, ';': 32,
        '(': 33, ')': 34, ':': 35, "'": 36, '!': 37, '?': 38
        # --------------------------
    },

    # Number of unique characters in the vocabulary.
    vocabulary_size=39,
)
