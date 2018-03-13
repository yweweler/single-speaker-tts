import tensorflow as tf


class Tacotron:
    def __init__(self, hparams):
        self.hparams = hparams

        # Create placeholders for the input data.
        x_mel_spec, y_linear_spec = self.inputs()

    def inputs(self):
        # Network inputs.
        x_mel_spec = tf.placeholder(dtype=tf.float32, shape=(None, None, self.hparams.n_mfcc), name='x_mel_spec')

        # Network target outputs for calculating the loss.
        y_linear_spec = tf.placeholder(dtype=tf.float32, shape=(None, None, self.hparams.n_mfcc), name='y_linear_spec')

        # x_mel_spec: shape=(N, T, n_mfccs)
        # y_linear_spec: shape=(N, T, 1 + n_fft // 2)
        return x_mel_spec, y_linear_spec

    def loss(self):
        raise NotImplementedError()

    def summary(self):
        raise NotImplementedError()

    def load(self, checkpoint_dir):
        raise NotImplementedError()
