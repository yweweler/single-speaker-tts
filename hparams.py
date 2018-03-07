import tensorflow as tf

# Default hyper-parameters:
hparams = tf.contrib.training.HParams(
    sampling_rate=16000,
    n_fft=1024,
    win_len=50.0,
    win_hop=12.5,
    n_mels=128,
    mel_fmin=0,
    mel_fmax=8000,
    n_mfcc=13
)
