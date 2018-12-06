import tensorflow as tf


# Default hyper-parameters:
dataset_params = tf.contrib.training.HParams(
    # Path to the dataset definition file.
    dataset_file='/tmp/LJSpeech-1.1/dataset.json',
)
