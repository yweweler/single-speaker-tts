import tensorflow as tf

# Default hyper-parameters:
hparams = tf.contrib.training.HParams(
    # Training parameters.
    train=tf.contrib.training.HParams(
        # Number of training epochs.
        n_epochs=5000,

        # Batch size used for training.
        batch_size=4,

        # Number of threads used to load data during training.
        n_threads=8,

        # Maximal number of samples to load from the train dataset.
        max_samples=250
    ),

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

    # Flag that controls application of the post-processing network.
    apply_post_processing=True,

    # Encoder network parameters.
    encoder=tf.contrib.training.HParams(
        # Embedding size for each sentence character.
        embedding_size=256,

        pre_net_layers=(
            # (units, dropout, activation).
            (256, 0.1, tf.nn.relu),
            (128, 0.1, tf.nn.relu)
        ),

        # Number of filter banks.
        n_banks=16,

        # Number of filters in each bank.
        n_filters=128,

        # Number of highway network layers.
        n_highway_layers=4,

        # Number of units in each highway layer.
        n_highway_units=128,

        projections=(
            # (filters, kernel_size, activation).
            (128, 3, tf.nn.relu),
            (128, 3, None)
        ),

        # Number of units in the encoder RNN.
        n_gru_units=128
    ),

    # Decoder network parameters.
    decoder=tf.contrib.training.HParams(
        pre_net_layers=(
            # (units, dropout, activation).
            (256, 0.1, tf.nn.relu),
            (128, 0.1, tf.nn.relu)
        ),

        # Number of decoder RNN layers.
        n_gru_layers=2,

        # Number of units in the decoder RNN.
        n_gru_units=256,

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

        # Number of highway network layers.
        n_highway_layers=4,

        # Number of units in each highway layer.
        n_highway_units=128,

        projections=(
            # (filters, kernel_size, activation).
            (256, 3, tf.nn.relu),
            (80, 3, None)
        ),

        # Number of units in the post-processing RNN.
        n_gru_units=128
    )
)
