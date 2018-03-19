import tensorflow as tf


def highway_network(inputs, units, activation=None, t_bias_init=-1.0):
    """

    Arguments:
        input:
        units:

    Returns:

    Notes:
        - See: http://arxiv.org/abs/1505.00387
        - We refer to T as the transform gate.
        - We refer to C as the carry gate, setting C = 1-T.
    """
    # TODO: Assert inputs.dim == outputs.dim == H(inputs) == T(inputs)

    # Initialization of kernel and bias are strongly dependant on the functional form of H.
    # In the paper they tested with activation functions ReLU and tanh.
    # They used transform gate biases between -1 and -10.
    # TODO: All other weights were initialized following the scheme introduced by (He et al., 2015).
    # TODO: Does the tacotron paper state anything on their exact configuration?

    h = tf.layers.dense(inputs=inputs,
                        units=units,
                        activation=activation,
                        kernel_initializer=tf.glorot_normal_initializer(),
                        bias_initializer=tf.zeros_initializer(),
                        name='H')

    t = tf.layers.dense(inputs=inputs,
                        units=units,
                        activation=tf.nn.sigmoid,  # See: "2.2 Training Deep Highway Networks"
                        # kernel_initializer=tf.glorot_normal_initializer(),
                        bias_initializer=tf.constant_initializer(t_bias_init),
                        # See: "2.2 Training Deep Highway Networks"
                        name='T')

    # t_bias_init = -1.0
    # ------------------------------------------------------------------------------------------------------------------
    # They found that a negative bias initialization was sufficient for learning to proceed in very deep networks for
    # various zero-mean initial distributions of W_H and different activation functions used by H.
    # (See: "2.2. Training Deep Highway Networks")

    return (h * t) + (inputs * (1.0 - t))
