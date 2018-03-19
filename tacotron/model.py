import tensorflow as tf


class Tacotron:
    def __init__(self, hparams, inputs):
        self.hparams = hparams

        # Create placeholders for the input data.
        self.inp_mel_spec, self.inp_linear_spec, self.seq_lengths = inputs
        self.pred_linear_spec = None

        self.model()

        self.loss_op = self.loss()

    def inputs(self):
        # Network inputs.
        inp_mel_spec = tf.placeholder(dtype=tf.float32,
                                      shape=(None, None, self.hparams.n_mels * self.hparams.reduction),
                                      name='inp_mel_spec')

        # Network target outputs for calculating the loss.
        inp_linear_spec = tf.placeholder(dtype=tf.float32,
                                         shape=(None, None, (1 + self.hparams.n_fft // 2) * self.hparams.reduction),
                                         name='inp_linear_spec')

        # inp_mel_spec: shape=(N, T//r, n_mels*r)
        # inp_linear_spec: shape=(N, T//r, (1 + n_fft // 2)*r)
        return inp_mel_spec, inp_linear_spec

    def get_inputs_placeholders(self):
        return self.inp_mel_spec, self.inp_linear_spec

    def postprocess(self, inputs):
        with tf.variable_scope('postprocess'):
            self.pred_linear_spec = tf.layers.dense(inputs=inputs,
                                                    units=(1 + self.hparams.n_fft // 2) * self.hparams.reduction,
                                                    activation=tf.nn.sigmoid,
                                                    kernel_initializer=tf.glorot_normal_initializer(),
                                                    bias_initializer=tf.glorot_normal_initializer())

        return self.pred_linear_spec

    def model(self):
        self.postprocess(self.inp_mel_spec)

        # self.pred_linear_spec = tf.Print(self.pred_linear_spec, [self.seq_lengths], summarize=32)

    def loss(self):
        # tf.losses.absolute_difference could be used either (in case reduction=Reduction.MEAN is used).
        return tf.reduce_mean(tf.abs(self.inp_linear_spec - self.pred_linear_spec))

    def get_loss_op(self):
        return self.loss_op

    def summary(self):
        tf.summary.scalar('loss', self.loss_op)

        # with tf.name_scope('reduced_inputs'):
        #     tf.summary.image('mel_spec', tf.expand_dims(self.inp_mel_spec, -1), max_outputs=1)
        #     tf.summary.image('linear_spec', tf.expand_dims(self.inp_linear_spec, -1), max_outputs=1)

        with tf.name_scope('normalized_inputs'):
            # tf.summary.image('mel_spec',
            #                  tf.expand_dims(
            #                      tf.reshape(self.inp_mel_spec[0],
            #                                 (1, -1, self.hparams.n_mels)), -1), max_outputs=1)

            tf.summary.image('linear_spec',
                             tf.expand_dims(
                                 tf.reshape(self.inp_linear_spec[0],
                                            (1, -1, (1 + self.hparams.n_fft // 2))), -1), max_outputs=1)

        # with tf.name_scope('reduced_outputs'):
        #     # tf.summary.image('inp_linear_spec', tf.expand_dims(self.inp_linear_spec, -1), max_outputs=1)
        #     tf.summary.image('linear_spec', tf.expand_dims(self.pred_linear_spec, -1), max_outputs=1)

        with tf.name_scope('normalized_outputs'):
            tf.summary.image('linear_spec',
                             tf.expand_dims(
                                 tf.reshape(self.pred_linear_spec[0],
                                            (1, -1, (1 + self.hparams.n_fft // 2))), -1), max_outputs=1)

        return tf.summary.merge_all()

    def load(self, checkpoint_dir):
        raise NotImplementedError()
