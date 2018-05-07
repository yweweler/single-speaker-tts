import tensorflow as tf

# Default hyper-parameters:
inference_params = tf.contrib.training.HParams(
    # Checkpoint folder used for loading the latest checkpoint.
    checkpoint_dir='/tmp/tacotron/pavoque/PAVOQUE',

    # Run folder to load a checkpoint from the checkpoint folder.
    checkpoint_load_run='train',

    # Run folder to save summaries in the checkpoint folder.
    checkpoint_save_run='inference',

    # The path were to save the inference results.
    synthesis_dir='/tmp/inference'
)
