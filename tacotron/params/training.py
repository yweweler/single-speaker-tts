import tensorflow as tf

from datasets.lj_speech import LJSpeechDatasetHelper

# Default hyper-parameters:
training_params = tf.contrib.training.HParams(
    # Number of training epochs.
    n_epochs=5000,

    # Batch size used for training.
    batch_size=4,

    # Number of threads used to load data during training.
    n_threads=8,

    # Folder containing the dataset.
    dataset_folder='/home/yves-noel/downloads/LJSpeech-1.1',

    # Dataset load helper.
    dataset_loader=LJSpeechDatasetHelper,

    # Maximal number of samples to load from the train dataset.
    max_samples=250,

    # Flag that enables/disables sample shuffle at the beginning of each epoch.
    shuffle_samples=False,

    # Number of batches to pre-calculate for feeding to the GPU.
    n_pre_calc_batches=8,

    # Number of samples each bucket can pre-fetch.
    n_samples_per_bucket=16,

    # Flag enabling the bucketing mechanism to output batches of smaller size than
    # `batch_size` if not enough samples are available.
    allow_smaller_batches=True,

    # Checkpoint folder used for training.
    checkpoint_dir='/tmp/tacotron/ljspeech_250_samples_completely_new',

    # Duration in seconds after which to save a checkpoint.
    checkpoint_save_secs=60 * 60,

    # Number of global steps after which to save the model summary.
    summary_save_steps=5,

    # Number of global steps after which to log the global steps per second.
    performance_log_steps=5,

    # The clipping ratio used for gradient clipping by global norm.
    gradient_clip_norm=5.0
)
