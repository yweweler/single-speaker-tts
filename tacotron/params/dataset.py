import tensorflow as tf

from datasets.lj_speech import LJSpeechDatasetHelper
from datasets.pavoque import PAVOQUEDatasetHelper

# Default hyper-parameters:
dataset_params = tf.contrib.training.HParams(
    # Folder containing the dataset.
    dataset_folder='/home/st/y/yw132854/workspace/pavoque',

    # Dataset load helper.
    dataset_loader=PAVOQUEDatasetHelper,

    # Vocabulary definition.
    # This definition has to include the padding and end of sequence tokens.
    # Both the padding and eos token should not be changed since they are used internally.
    vocabulary_dict={
        'pad': 0,  # padding
        'eos': 1,  # end of sequence
        # --------------------------
        'i': 2, 'n': 3, ' ': 4, 's': 5, 'e': 6, 'r': 7, 'j': 8, 'u': 9, 'g': 10, 'd': 11, 'a': 12,
        'b': 13, 't': 14, 'c': 15, 'h': 16, 'l': 17, 'ä': 18, '.': 19, 'ü': 20, 'm': 21, 'p': 22,
        'w': 23, 'z': 24, ',': 25, 'ö': 26, 'o': 27, 'f': 28, 'k': 29, ';': 30, 'y': 31, 'v': 32,
        'x': 33, 'ß': 34, ':': 35, 'q': 36, '"': 37, '?': 38, '!': 39, "'": 40, '/': 41
        # --------------------------
    },

    # Number of unique characters in the vocabulary.
    vocabulary_size=42,
)
