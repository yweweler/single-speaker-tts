import tensorflow as tf

# Default hyper-parameters:
hparams = tf.contrib.training.HParams(
    # Target sampling rate.
    sampling_rate=16000,

    # Number of unique characters in the vocabulary.
    vocabulary_size=63,

    # FFT window size.
    n_fft=1024,

    # Windows length in ms.
    # win_len=25.0,
    win_len=50.0,

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
    reduction=1,

    # Flag that controls application of the post-processing network.
    apply_post_processing=False,

    # Encoder network parameters.
    encoder=tf.contrib.training.HParams(
        embedding_size=256,

        pre_net_layers=(
            # (units, dropout, activation).
            (256, 0.5, tf.nn.relu),
            (128, 0.5, tf.nn.relu)
        ),

        n_banks=16,
        n_filters=128,
        n_highway_layers=4,
        n_highway_units=128,
        projections=(
            # (filters, kernel_size, activation).
            (128, 3, tf.nn.relu),
            (128, 3, None)
        ),
        n_gru_units=128
    ),

    # Decoder network parameters.
    decoder=tf.contrib.training.HParams(
        pre_net_layers=(
            # (units, dropout, activation).
            (256, 0.5, tf.nn.relu),
            (128, 0.5, tf.nn.relu)
        ),

        n_gru_layers=2,
        # TODO: 256 in the paper.
        n_gru_units=128,
        target_size=80
    ),

    # Post-processing network parameters.
    post=tf.contrib.training.HParams(
        n_banks=8,
        n_filters=128,
        n_highway_layers=4,
        n_highway_units=128,
        projections=(
            # (filters, kernel_size, activation).
            (256, 3, tf.nn.relu),
            (80, 3, None)
        ),
        n_gru_units=128
    )
)
