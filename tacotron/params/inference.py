import tensorflow as tf

# Default hyper-parameters:
inference_params = tf.contrib.training.HParams(
    # Checkpoint folder used for loading the latest checkpoint.
    checkpoint_dir='/tmp/tacotron/ljspeech_all',

    # Run folder to load a checkpoint from the checkpoint folder.
    checkpoint_load_run='train',

    # The path were to save the inference results.
    synthesis_path='/tmp/inference'
)
