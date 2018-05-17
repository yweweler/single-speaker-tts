import tensorflow as tf

from datasets.lj_speech import LJSpeechDatasetHelper

# Default hyper-parameters:
training_params = tf.contrib.training.HParams(
    # Number of training epochs.
    n_epochs=5000,

    # Batch size used for training.
    batch_size=4,

    # Number of threads used to load data during training.
    n_threads=4,

    # Maximal number of samples to load from the train dataset.
    max_samples=None,

    # Flag that enables/disables sample shuffle at the beginning of each epoch.
    shuffle_samples=False,

    # Flag telling the training code to load pre-processed features or calculate them on the fly.
    load_preprocessed=True,

    # Number of batches to pre-calculate for feeding to the GPU.
    n_pre_calc_batches=8,

    # Number of samples each bucket can pre-fetch.
    n_samples_per_bucket=16,

    # The number of buckets to create. Note that this is the number of buckets that are actually
    # created. If less buckets are needed for proper sorting of the data, less buckets are used.
    n_buckets=20,

    # Flag enabling the bucketing mechanism to output batches of smaller size than
    # `batch_size` if not enough samples are available.
    allow_smaller_batches=False,

    # Checkpoint folder used for training.
    checkpoint_dir='/tmp/tacotron/pavoque/PAVOQUE',

    # Run folder to load data from and save data in to the checkpoint folder.
    checkpoint_run='train',

    # Duration in seconds after which to save a checkpoint.
    checkpoint_save_secs=60 * 30,

    # Number of global steps after which to save the model summary.
    summary_save_steps=50,

    # Flag controlling whether to actually write a write a summary during training.
    # The only exceptions to this are the attention alignment plots and the train losses.
    write_summary=False,

    # Maximal number of checkpoints to keep.
    checkpoints_to_keep=3000,

    # Number of global steps after which to log the global steps per second.
    performance_log_steps=50,

    # The clipping ratio used for gradient clipping by global norm.
    gradient_clip_norm=5.0,

    # Initial learning rate.
    lr=0.001,

    # Number of global steps after which the learning rate should be decayed.
    lr_decay_steps=25000,

    # Learning rate decay rate.
    lr_decay_rate=0.9,

    # Flag controlling if the learning rate decay should be staircase or not.
    lr_staircase=True
)
