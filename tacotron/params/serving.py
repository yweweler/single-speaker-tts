import tensorflow as tf

# Default hyper-parameters:
serving_params = tf.contrib.training.HParams(
    # Directory to load the exported model from.
    export_dir='/tmp/export/ljspeech',

    # Version number of the exported model to be loaded.
    export_version=1,

)
