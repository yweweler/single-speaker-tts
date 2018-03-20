import tensorflow as tf


def highway_network_layer(inputs, units, activation=tf.nn.relu, t_bias_init=-1.0):
    """
    Implementation of a Highway Network layer.

    See: http://arxiv.org/abs/1505.00387

    Arguments:
        inputs (tf.Tensor):
            The shape is expected to be shape=(B, T, F) with B being the batch size, T being the
            number of time frames and F being the size of the features.

        units (int):
            The number of units in the highway layer.
            The number of units is expected to match the feature size F.

        activation (:obj:`function`, optional):
            pass

        t_bias_init (:obj:`float`, optional):
            pass

    Returns:
        tf.Tensor:
            pass

    Notes:
        - See [1]: Highway Networks, http://arxiv.org/abs/1505.00387
        - See [2]: http://arxiv.org/abs/1502.01852

        The Tacotron paper only states that they are able to train their architecture using
        "random initialization". However, it is unclear to me if they are using a plain Xavier
        initialization or initialization based on (He et al., 2015) [2] as described in the
        Highway Networks paper.

        The procedure from [2] takes into account that the ReLU and PReLU activations are not
        linear, as opposed to Xavier which assumes that activations are linear.
        However, considering the measurements described in [2] there seems to be no superior
        solution when using smaller networks instead of very deep networks.

        As described in [1], section "2.2. Training Deep Highway Networks", a negative transform
        gate bias initialization was sufficient for learning to proceed in very deep networks for
        various zero-mean initial distributions of W_H and different activation functions used by
        H.

        Since Xavier initialization is zero-mean and the Highway Networks used in Tacotron are
        not particularly deep there should not be a major impact from using it.
    """
    # Make sure the the input dimensionality is equal to the output dimensionality.
    tf.assert_equal(inputs.shape[-1], units)

    h = tf.layers.dense(inputs=inputs,
                        units=units,
                        activation=activation,
                        kernel_initializer=tf.glorot_normal_initializer(),
                        bias_initializer=tf.zeros_initializer(),
                        name='H')

    # For the transform gate we follow [1], section "2.2 Training Deep Highway Networks" using a
    # sigmoid activation function and a negative bias initializer.
    t = tf.layers.dense(inputs=inputs,
                        units=units,
                        activation=tf.nn.sigmoid,
                        kernel_initializer=tf.glorot_normal_initializer(),
                        bias_initializer=tf.constant_initializer(t_bias_init),
                        name='T')

    # TODO: For some reason pycharm thinks that this is a plain floating point operation.
    return (h * t) + (inputs * (1.0 - t))
