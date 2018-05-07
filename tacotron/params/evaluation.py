import tensorflow as tf

# Default hyper-parameters:
evaluation_params = tf.contrib.training.HParams(
    # Batch size used for evaluation.
    batch_size=4,

    # Number of threads used to load data during evaluation.
    n_threads=4,

    # Maximal number of samples to load from the evaluation dataset.
    max_samples=256,

    # Flag that enables/disables sample shuffle at the beginning of each epoch.
    shuffle_samples=False,

    # Number of batches to pre-calculate for feeding to the GPU.
    n_pre_calc_batches=8,

    # Number of samples each bucket can pre-fetch.
    n_samples_per_bucket=16,

    # The number of buckets to create. Note that this is the number of buckets that are actually
    # created. If less buckets are needed for proper sorting of the data, less buckets are used.
    n_buckets=20,

    # Flag enabling the bucketing mechanism to output batches of smaller size than
    # `batch_size` if not enough samples are available.
    allow_smaller_batches=True,

    # Checkpoint folder used for loading the latest checkpoint.
    checkpoint_dir='/tmp/tacotron/pavoque',

    # Run folder to load a checkpoint from the checkpoint folder.
    checkpoint_load_run='train',

    # Run folder to save summaries in the checkpoint folder.
    checkpoint_save_run='evaluate',

    # Number of global steps after which to save the model summary.
    summary_save_steps=50,

    # Number of global steps after which to log the global steps per second.
    performance_log_steps=50
)
