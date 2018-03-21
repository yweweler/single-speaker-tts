import tensorflow as tf
from theano.gof.graph import inputs


def prelu(inputs, layer_wise=True):
    """
    Implements a Parametric Rectified Linear Unit (PReLU).

    See: https://arxiv.org/abs/1502.01852

    Arguments:
        inputs (tf.Tensor):
            An input tensor to be activated.

        layer_wise (boolean):
            If True (default), only creates one trainable activation coefficient `alpha` for all
            elements of the input. If False a separate activation coefficient `alpha` is created
            for each element of the last dimension of the input tensor. Resulting in `alpha`
            being a vector with `inputs.shape[-1]` elements.

    Returns:
        tf.Tensor:
            Activation of the input tensor.
    """
    # TODO: Can this also be written as tf.maximum/minimum(0.0, inputs)?
    # TODO: Can this otherwise be written as zeros = tf.constant(0.0, shape=inputs.shape)?
    # TODO: Is there a noticeable performance difference between these implementations?

    # Measurements:
    #   - zeros = tf.zeros_like(inputs)                         => bs=8, epochs=5: 773s, err: ~0.034
    #   - zeros = tf.constant(0.0, [shape=inputs.shape[-1]])    => bs=8, epochs=5: 737s, err: ~0.034
    #   - tf.maximum/minimum(0.0, inputs)                       => bs=8, epochs=5: 737s, err: ~0.034
    #   - plain ReLU                                            => bs=8, epochs=5: 671s, err: ~0.035

    zeros = tf.constant(value=0.0, shape=[inputs.shape[-1]])

    if layer_wise:
        alpha_shape = 1
    else:
        alpha_shape = inputs.shape[-1]

    alpha = tf.get_variable('alpha',
                            shape=alpha_shape,
                            initializer=tf.constant_initializer(0.01))

    tf.summary.histogram('alpha', alpha)

    # return tf.maximum(zeros, inputs) + alpha * tf.minimum(zeros, inputs)
    return tf.maximum(zeros, inputs) + alpha * tf.minimum(zeros, inputs)


def highway_network(inputs, units, layers, scope, activation=tf.nn.relu):
    """
    Implementation of a multi layer Highway Network.

    See: http://arxiv.org/abs/1505.00387

    Arguments:
        inputs (tf.Tensor):
            The shape is expected to be shape=(B, T, F) with B being the batch size, T being the
            number of time frames and F being the size of the features.

        units (int):
            The number of units in the highway layers.
            Units is expected to fulfill `units` == F.

        layers (int):
            The number of highway layers to stack.

        scope (str):
            Tensorflow variable scope to wrap the layers in.

        activation (:obj:`function`, optional):
            Activation function for each of the fully connected layers.

    Returns:
        tf.Tensor:
            Tensor of shape shape=(B, T, F) with B being the batch size, T being the
            number of time frames and F being the size of the features.

    """
    # Make sure the the input dimensionality is equal to the output dimensionality.
    tf.assert_equal(inputs.shape[-1], units)

    network = inputs
    with tf.variable_scope(scope):
        for layer in range(layers):
            network = highway_network_layer(inputs=network, units=units, activation=activation,
                                            scope='highway_layer_{}'.format(layer))

    return network


def highway_network_layer(inputs, units, scope, activation=tf.nn.relu, t_bias_init=-1.0):
    """
    Implementation of a Highway Network layer.

    See: http://arxiv.org/abs/1505.00387

    Arguments:
        inputs (tf.Tensor):
            The shape is expected to be shape=(B, T, F) with B being the batch size, T being the
            number of time frames and F being the size of the features.

        units (int):
            The number of units in the highway layer.
            Units is expected to fulfill `units` == F.

        scope (str):
            Tensorflow variable scope to construct the layer in.

        activation (:obj:`function`, optional):
            Activation function for the fully connected layer H.

        t_bias_init (:obj:`float`, optional):
            Constant value for initializing the transform gate bias of the transform gate T.

    Returns:
        tf.Tensor:
            Tensor of shape shape=(B, T, F) with B being the batch size, T being the
            number of time frames and F being the size of the features.

    Notes:
        - See [1]: Highway Networks, https://arxiv.org/abs/1505.00387
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
    with tf.variable_scope(scope):
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
