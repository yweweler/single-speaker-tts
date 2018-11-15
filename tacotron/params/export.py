import tensorflow as tf

# Default hyper-parameters:
export_params = tf.contrib.training.HParams(
    # Checkpoint folder used for loading the latest checkpoint.
    checkpoint_dir='/tmp/checkpoints/ljspeech',

    # Run folder to load a checkpoint from the checkpoint folder.
    checkpoint_load_run='train',

    # Direct path to a checkpoint file to restore for export.
    # If `checkpoint_file` is None, the latest checkpoint will be restored.
    checkpoint_file=None,

    # Directory to save the exported model in.
    export_dir='/tmp/export/ljspeech',

    # Version number to export the model under.
    export_version=1,

)
