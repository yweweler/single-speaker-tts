import tensorflow as tf


def prelu(inputs, layer_wise=False):
    """
    Implements a Parametric Rectified Linear Unit (PReLU).

    See: https://arxiv.org/abs/1502.01852

    Arguments:
        inputs (tf.Tensor):
            An input tensor to be activated.

        layer_wise (boolean):
            If True, only creates one trainable activation coefficient `alpha` for all
            elements of the input. If False (default) a separate activation coefficient `alpha` is
            created for each element of the last dimension of the input tensor. Resulting in `alpha`
            being a vector with `inputs.shape[-1]` elements.

    Returns:
        tf.Tensor:
            Activation of the input tensor.
    """
    zeros = tf.constant(value=0.0, shape=[inputs.shape[-1]])

    if layer_wise:
        alpha_shape = 1
    else:
        alpha_shape = inputs.shape[-1]

    alpha = tf.get_variable('alpha',
                            shape=alpha_shape,
                            initializer=tf.constant_initializer(0.01))

    tf.summary.histogram('alpha', alpha)

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


def pre_net(inputs, units=(256, 128), dropout=(0.5, 0.5), scope='pre_net', training=True):
    """
    Implementation of the pre-net described in "Tacotron: Towards End-to-End Speech Synthesis".

    See: https://arxiv.org/abs/1703.10135

    Arguments:
        inputs (tf.Tensor):
            The shape is expected to be shape=(B, T, F) with B being the batch size, T being the
            number of time frames and F being the size of the features.

        units (:obj:`list` of int):
            A list of length L defining the number of units for L layers.
            Defaults to (256, 128).

        dropout (:obj:`list` of float):
            A list of length L defining the dropout rates for L layers.
            Defaults to (0.5, 0.5).

        scope (str):
            Tensorflow variable scope to wrap the layers in.

        training (boolean):
            Boolean defining whether to apply the dropout or not.
            Default is True.

    Returns:
        tf.Tensor:
            A tensor which shape is expected to be shape=(B, T, units[-1]) with B being the batch
            size, T being the number of time frames.
    """
    assert (len(units) == len(dropout)), 'The number of supplied units does not match the number ' \
                                         'of supplied dropout rates.'

    network = inputs
    with tf.variable_scope(scope):
        for layer_units, layer_dropout in zip(units, dropout):
            network = tf.layers.dense(inputs=network,
                                      units=layer_units,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.glorot_normal_initializer(),
                                      bias_initializer=tf.zeros_initializer(),
                                      name='FC-{}-ReLU'.format(layer_units))

            network = tf.layers.dropout(inputs=network,
                                        rate=layer_dropout,
                                        training=training,
                                        name='dropout')

        return network


def conv_1d_filter_banks(inputs, n_banks, n_filters, activation=tf.nn.relu, training=True):
    # TODO: Add documentation.

    # K := Number of filter banks. (n_banks)
    # C_k := Number of filters in the K-th filter bank. (n_filters)

    # See: section "3.1 CBHG Module"
    # "The input sequence is first convolved with K sets of 1-D convolutional filters, where the
    # k-th set contains C_k filters of width k (i.e. k = 1, 2, ... , K)."
    filter_banks = []
    for bank in range(n_banks):
        # See: section "3.1 CBHG Module"
        # "Note that we use a stride of 1 to preserve the original time resolution."
        # ------------------------------------------------------------------------------------------
        # Each conv1d bank will produce an output with shape=(B, T, n_filters).
        filter_bank = tf.layers.conv1d(inputs=inputs,
                                       filters=n_filters,
                                       kernel_size=bank + 1,
                                       strides=1,
                                       activation=activation,
                                       padding='SAME')

        # TODO: Compare the generation performance between the two BN approaches?
        # Note: The Tacotron paper is not clear at this point. One could either apply BN K times
        # to each filter bank and concatenate them or concatenate them and apply BN once.
        # Since they state "Batch normalization (Ioffe & Szegedy, 2015) is used for all
        # convolutional layers." I will apply BN before concatenation.
        # TODO: read the batch normalization paper: http://arxiv.org/abs/1502.03167
        # filter_bank = tf.layers.batch_normalization(inputs=filter_bank,
        #                                             training=training,
        #                                             # name='batch_normalization',
        #                                             # renorm=True,  # TODO: read the corresp. paper.
        #                                             fused=True,     # TODO: Speed improvement?
        #                                             scale=False     # TODO: Is the follow. relu
        #                                             # proj. layer enough?
        #                                             )
        # TODO: Read renorm paper: https://arxiv.org/abs/1702.03275

        filter_banks.append(filter_bank)

    # See: section "3.1 CBHG Module"
    # "The convolution outputs are stacked together and further max pooled along time to increase
    # local invariances."
    # ------------------------------------------------------------------------------------------
    # Stacking the filter outputs for each spectrogram frame produces an output
    # with shape=(B, T, C_k * K).
    return tf.concat(filter_banks, axis=-1)

    # See: section "3.1 CBHG Module"
    # We further pass the processed sequence to a few fixed-width 1-D convolutions, whose outputs
    # are added with the original input sequence via residual connections (He et al., 2016).
