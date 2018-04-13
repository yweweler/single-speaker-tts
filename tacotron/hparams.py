import tensorflow as tf

from datasets.lj_speech import LJSpeechDatasetHelper

# Default hyper-parameters:
hparams = tf.contrib.training.HParams(
    # Vocabulary definition.
    # This definition has to include the padding and end of sequence tokens.
    # Both the padding and eos token should not be changed since they are used internally.
    vocabulary_dict={
        'pad': 0,  # padding
        'eos': 1,  # end of sequence
        # --------------------------
        'p': 2,
        'r': 3,
        'i': 4,
        'n': 5,
        't': 6,
        'g': 7,
        ' ': 8,
        'h': 9,
        'e': 10,
        'o': 11,
        'l': 12,
        'y': 13,
        's': 14,
        'w': 15,
        'c': 16,
        'a': 17,
        'd': 18,
        'f': 19,
        'm': 20,
        'x': 21,
        'b': 22,
        'v': 23,
        'u': 24,
        'k': 25,
        'j': 26,
        'z': 27,
        'q': 28,
        # --------------------------
    },

    # Target sampling rate.
    sampling_rate=22050,

    # Number of unique characters in the vocabulary.
    vocabulary_size=29,

    # FFT window size.
    n_fft=1024,

    # Windows length in ms.
    # win_len=25.0,
    win_len=45.0,

    # Window stride in ms.
    # win_hop=8.0,
    win_hop=12.5,

    # Number of Mel bands to generate.
    n_mels=80,

    # Mel spectrum lower cutoff frequency.
    mel_fmin=0,

    # Mel spectrum upper cutoff frequency.
    mel_fmax=8000,

    # Number of Mel-frequency cepstral coefficients to generate.
    n_mfcc=13,

    # Tacotron reduction factor r.
    reduction=3,

    # Training parameters.
    train=tf.contrib.training.HParams(
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
        shuffle_smaples=False,

        # Number of batches to pre-calculate for feeding to the GPU.
        n_pre_calc_batches=8,

        # Number of samples each bucket can pre-fetch.
        n_samples_per_bucket=16,

        # Flag enabling the bucketing mechanism to output batches of smaller size than
        # `batch_size` if not enough samples are available.
        allow_smaller_batches=True,

        # Checkpoint folder used for training.
        checkpoint_dir='/tmp/tacotron/ljspeech_250_samples',

        # Duration in seconds after which to save a checkpoint.
        checkpoint_save_secs=60 * 60,

        # Number of global steps after which to save the model summary.
        summary_save_steps=5,

        # Number of global steps after which to log the global steps per second.
        performance_log_steps=5,

        # The clipping ratio used for gradient clipping by global norm.
        gradient_clip_norm=5.0
    ),

    # Flag that controls application of the post-processing network.
    apply_post_processing=False,

    # Encoder network parameters.
    encoder=tf.contrib.training.HParams(
        # Embedding size for each sentence character.
        embedding_size=256,

        pre_net_layers=(
            # (units, dropout, activation).
            (256, 0.5, tf.nn.relu),
            (128, 0.5, tf.nn.relu)
        ),

        # Number of filter banks.
        n_banks=16,

        # Number of filters in each bank.
        n_filters=128,

        projections=(
            # (filters, kernel_size, activation).
            (128, 3, tf.nn.relu),
            (128, 3, None)
        ),

        # Number of highway network layers.
        n_highway_layers=4,

        # Number of units in each highway layer.
        n_highway_units=128,

        # Number of units in the encoder RNN.
        n_gru_units=128
    ),

    # Decoder network parameters.
    decoder=tf.contrib.training.HParams(
        pre_net_layers=(
            # (units, dropout, activation).
            (256, 0.5, tf.nn.relu),
            (128, 0.5, tf.nn.relu)
        ),

        # Number of decoder RNN layers.
        n_gru_layers=2,

        # Number of units in the decoder RNN.
        n_decoder_gru_units=256,

        # Number of units in the attention RNN.
        n_attention_units=256,

        # Dimensionality of a single RNN target frame.
        target_size=80,

        # Maximum number of decoder iterations for evaluation and inference.
        maximum_iterations=1000,
    ),

    # Post-processing network parameters.
    post=tf.contrib.training.HParams(
        # Number of filter banks.
        n_banks=8,

        # Number of filters in each bank.
        n_filters=128,

        projections=(
            # (filters, kernel_size, activation).
            (256, 3, tf.nn.relu),
            (80, 3, None)
        ),

        # Number of highway network layers.
        n_highway_layers=4,

        # Number of units in each highway layer.
        n_highway_units=128,

        # Number of units in the post-processing RNN.
        n_gru_units=128
    )
)
