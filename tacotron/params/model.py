"""
Contains the model definition.
"""

import tensorflow as tf

# Default hyper-parameters:
from tensorflow.contrib.seq2seq import BahdanauAttention, LuongAttention

from tacotron.attention import AttentionMode, AttentionScore, LocalLuongAttention

model_params = tf.contrib.training.HParams(
    # Number of unique characters in the vocabulary.
    vocabulary_size=39,

    # Target sampling rate.
    sampling_rate=22050,

    # FFT window size.
    n_fft=2048,

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
    reduction=5,

    # Flag that controls application of the post-processing network.
    apply_post_processing=True,

    # Linear scale magnitudes are raise to the power of `magnitude_power` before reconstruction.
    magnitude_power=1.3,

    # The number of Griffin-Lim reconstruction iterations.
    reconstruction_iterations=50,

    # Flag allowing to force the use accelerated RNN implementation from CUDNN.
    force_cudnn=False,

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

        # Maximum number of decoder iterations after which to stop for evaluation and inference.
        # This is equal to the number of mel-scale spectrogram frames generated.
        maximum_iterations=1000,
    ),

    # Attention parameters.
    attention=tf.contrib.training.HParams(
        # mechanism=BahdanauAttention,
        mechanism=LuongAttention,
        # mechanism=LocalLuongAttention,

        # Luong local style content based scoring function.
        luong_local_score=AttentionScore.DOT,

        # Luong local style attention mode.
        luong_local_mode=AttentionMode.MONOTONIC,

        # Luong local: Force a gaussian distribution onto the scores in the attention window.
        luong_force_gaussian=True,

        # Luong local style window D parameter. (Window size will be `2D+1`).
        luong_local_window_D=10
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
