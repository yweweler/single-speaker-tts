import tensorflow as tf


class Tacotron:
    def __init__(self, hparams, inputs):
        self.hparams = hparams

        # Create placeholders for the input data.
        self.inp_mel_spec, self.inp_linear_spec = inputs
        self.pred_linear_spec = None

        self.model()

        self.loss_op = self.loss()

    def inputs(self):
        # Network inputs.
        inp_mel_spec = tf.placeholder(dtype=tf.float32, shape=(None, None, self.hparams.n_mels), name='inp_mel_spec')

        # Network target outputs for calculating the loss.
        inp_linear_spec = tf.placeholder(dtype=tf.float32, shape=(None, None, 1 + self.hparams.n_fft // 2),
                                         name='inp_linear_spec')

        # inp_mel_spec: shape=(N, T, n_mfccs)
        # inp_linear_spec: shape=(N, T, 1 + n_fft // 2)
        return inp_mel_spec, inp_linear_spec

    def get_inputs_placeholders(self):
        return self.inp_mel_spec, self.inp_linear_spec

    def model(self):
        self.pred_linear_spec = tf.layers.dense(inputs=self.inp_mel_spec, units=1 + self.hparams.n_fft // 2,
                                                activation=tf.nn.sigmoid,
                                                kernel_initializer=tf.glorot_normal_initializer(),
                                                bias_initializer=tf.glorot_normal_initializer())

    def loss(self):
        # tf.losses.absolute_difference could be used either (in case reduction=Reduction.MEAN is used).
        return tf.reduce_mean(tf.abs(self.inp_linear_spec - self.pred_linear_spec))

    def get_loss_op(self):
        return self.loss_op

    def summary(self):
        tf.summary.scalar('loss', self.loss())
        tf.summary.image('inp_mel_spec', tf.expand_dims(self.inp_mel_spec, -1), max_outputs=1)
        with tf.name_scope('linear_spec'):
            tf.summary.image('inp_linear_spec', tf.expand_dims(self.inp_linear_spec, -1), max_outputs=1)
            tf.summary.image('pred_linear_spec', tf.expand_dims(self.pred_linear_spec, -1), max_outputs=1)
            tf.summary.image('error_linear_spec', tf.expand_dims(self.inp_linear_spec - self.pred_linear_spec, -1),
                             max_outputs=1)

        return tf.summary.merge_all()

    def load(self, checkpoint_dir):
        raise NotImplementedError()
