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
            Default activation function is `tf.nn.relu`.

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

    return tf.add(tf.multiply(h, t), tf.multiply(inputs, (1.0 - t)))
    # return h * t + inputs * (1.0 - t)


def pre_net(inputs, layers, scope='pre_net', training=True):
    """
    Implementation of the pre-net described in "Tacotron: Towards End-to-End Speech Synthesis".

    See: https://arxiv.org/abs/1703.10135

    Arguments:
        inputs (tf.Tensor):
            The shape is expected to be shape=(B, T, F) with B being the batch size, T being the
            number of time frames and F being the size of the features.

        layers (:obj:`list` of :obj:`tuple`):
            A list of length L containing tuples of the form (units, dropout, activation ) defining
            the number of units the dropout rate and the activation function for L layers.

        scope (str):
            Tensorflow variable scope to wrap the layers in.

        training (boolean):
            Boolean defining whether to apply the dropout or not.
            Default is True.

    Returns:
        tf.Tensor:
            A tensor which shape is expected to be shape=(B, T, layers[-1].units) with B being
            the batch size, T being the number of time frames.
    """
    network = inputs
    with tf.variable_scope(scope):
        for i, (layer_units, layer_dropout, layer_activation) in enumerate(layers):
            network = tf.layers.dense(inputs=network,
                                      units=layer_units,
                                      activation=layer_activation,
                                      kernel_initializer=tf.glorot_normal_initializer(),
                                      bias_initializer=tf.zeros_initializer(),
                                      name='{}-FC-{}'.format(i + 1, layer_units))

            network = tf.layers.dropout(inputs=network,
                                        rate=layer_dropout,
                                        training=training,
                                        name='dropout')

        return network


def conv_1d_filter_banks(inputs, n_banks, n_filters, scope, activation=tf.nn.relu, training=True):
    """
    Implementation of a 1D convolutional filter banks described in "Tacotron: Towards End-to-End
    Speech Synthesis".

    See: "Tacotron: Towards End-to-End Speech Synthesis"
      * Source: [1] https://arxiv.org/abs/1703.10135

    The Tacotron paper is the main source for the implementation of the filter banks.

    See: "Fully Character-Level Neural Machine Translation without Explicit Segmentation"
      * Source: [2] https://arxiv.org/abs/1610.03017

    [1] references [2] as the bases the CBHG (1-D convolution bank + highway network +
    bidirectional GRU) concept was derived from. Especially Figure 2. of [2] section 4.1 gives a
    nice overview on how the convolutional filters are applied.

    Arguments:
        inputs (tf.Tensor):
            The shape is expected to be shape=(B, T, F) with B being the batch size, T being the
            number of time frames and F being the size of the features.

        n_banks (int):
            The number of filter banks to use.

        n_filters (int):
            The dimensionality of the output space of each filter bank (i.e. the number of
            filters in each bank).

        scope (str):
            Tensorflow variable scope to construct the layer in.

        activation (:obj:`function`, optional):
            Activation function for the filter banks.
            Default activation function is `tf.nn.relu`.

        training (boolean):
            Boolean defining whether to apply the batch normalization or not.
            Default is True.

    Returns:
        tf.Tensor:
            A tensor which shape is expected to be shape=(B, T, n_banks * n_filters) with B being
            the batch size, T being the number of time frames.
    """
    with tf.variable_scope(scope):
        # [1], section 3.1 CBHG Module:
        # "The input sequence is first convolved with K sets of 1-D convolutional filters, where the
        # k-th set contains C_k filters of width k (i.e. k = 1, 2, ... , K)."
        filter_banks = []
        for bank in range(n_banks):
            # [1], section 3.1 CBHG Module:
            # "Note that we use a stride of 1 to preserve the original time resolution."
            # filter_bank.shape => (B, T, n_filters)
            filter_bank = tf.layers.conv1d(inputs=inputs,
                                           filters=n_filters,
                                           kernel_size=bank + 1,
                                           strides=1,
                                           activation=activation,
                                           padding='SAME',
                                           name='conv-{}-{}'.format(bank + 1, n_filters))

            # Improvement: In my opinion the Tacotron paper is not clear on how to apply batch
            # normalization on the filter banks.
            #
            # I see two ways we can apply BN here:
            #   1. Apply a separate BN operation to the K filter banks and concatenate the output.
            #   2. Concatenate the K filter banks and apply a single BN operation.
            #
            # As [1] states: "Batch normalization (Ioffe & Szegedy, 2015) is used for all
            # convolutional layers.", I have decided to implement case 1.

            # Improvement: What would be the effect of setting renorm=True?
            filter_bank = tf.layers.batch_normalization(inputs=filter_bank,
                                                        training=training,
                                                        fused=True,
                                                        scale=False)

            # filter_bank.shape => (B, T, n_filters)
            filter_banks.append(filter_bank)

        # [1], section 3.1 CBHG Module: "The convolution outputs are stacked together [...]"
        # shape=(B, T, n_banks * n_filters)
        stacked_banks = tf.concat(filter_banks, axis=-1)

    return stacked_banks


def conv_1d_projection(inputs, n_filters, kernel_size, activation, scope, training=True):
    """
    Implementation of a 1D convolution projection described in "Tacotron: Towards End-to-End Speech
    Synthesis".

    See: https://arxiv.org/abs/1703.10135

    The purpose of this operation is to project the fed data from the entered dimensionality
    in the last rank (F) into the dimensionality 'n_filters' using 1D convolution filters.

    Arguments:
        inputs (tf.Tensor):
            The shape is expected to be shape=(B, T, F) with B being the batch size, T being the
            number of time frames and F being the size of the features.

        n_filters (int):
            The dimensionality of the output space (i.e. the number of filters in the convolution).

        kernel_size (int):
            Length of the 1D convolution window.

        activation:
            Activation function. Set it to None to maintain a linear activation.

        scope (str):
            Tensorflow variable scope to wrap the layers in.

        training (boolean):
            Boolean defining whether to apply the batch normalization or not.
            Default is True.

    Returns:
        tf.Tensor:
            A tensor which shape is expected to be shape=(B, T, n_filters) with B being the batch
            size, T being the number of time frames.
    """
    with tf.variable_scope(scope):
        network = tf.layers.conv1d(inputs=inputs,
                                   filters=n_filters,
                                   kernel_size=kernel_size,
                                   strides=1,
                                   activation=activation,
                                   padding='SAME')

        # Improvement: What would be the effect of setting renorm=True?
        network = tf.layers.batch_normalization(inputs=network,
                                                training=training,
                                                fused=True,
                                                scale=True)

    return network


def cbhg(inputs, n_banks, n_filters, n_highway_layers, n_highway_units, projections,
         n_gru_units, training=True):
    """
    Implementation of a CBHG (1-D convolution bank + highway network + bidirectional GRU)
    described in "Tacotron: Towards End-to-End Speech Synthesis".

    See: "Tacotron: Towards End-to-End Speech Synthesis"
      * Source: [1] https://arxiv.org/abs/1703.10135

    Arguments:
        inputs (tf.Tensor):
            The shape is expected to be shape=(B, T, F) with B being the batch size, T being the
            number of time frames and F being the size of the features.

        n_banks (int):
            The number of 1D convolutional filter banks to use.

        n_filters (int):
            The dimensionality of the output space of each 1d convolutional filter bank (i.e. the
            number of filters in each bank).

        n_highway_layers (int):
            The number of highway layers to stack in the highway network.

        n_highway_units (int):
            The number of units to use for the highway network layers.
            Note that the result of the residual connection with shape=(B, T, F) is projected
            using a dense network to match required shape=(B, T, n_highway_units) for the highway
            network.

        projections (:obj:`list` of :obj:`tuple`):
            A list containing parameter tuples of the form (filters, kernel_size, activation)
            for each 1D convolutional projection layer to be created.
            `filters` being the dimensionality of the output space of the 1d convolutional
            projection (i.e. the number of filters in the projection layer).
            `kernel_size` being the kernel size of the 1D convolution.
            `activation` being the used activation function for the projection layer.

        n_gru_units (int):
            The number of units to use for the bi-direction GRU.

        training (boolean):
            Boolean defining whether the network will be trained or just used for inference.
            Default is True.

    Returns:
        (outputs, output_states):
            outputs (tf.Tensor): The output states (output_fw, output_bw) of the RNN concatenated
                over time. A tensor which shape is expected to be shape=(B, T, n_gru_units * 2)
                with B being the batch size, T being the number of time frames.

            output_states (tf.Tensor): A tensor containing the forward and the backward final states
                (output_state_fw, output_state_bw) of the bidirectional rnn.
                Its shape is expected to be shape=(2, n_gru_units).
    """
    # network.shape => (B, T, n_banks * n_filters)
    network = conv_1d_filter_banks(inputs=inputs,
                                   n_banks=n_banks,
                                   n_filters=n_filters,
                                   scope='convolution_banks',
                                   training=training)

    # [1], section 3.1 CBHG Module:
    # "The convolution outputs are [...] further max pooled along time to increase
    #  local invariances."

    # network.shape => (B, T, n_banks * n_filters)
    network = tf.layers.max_pooling1d(inputs=network,
                                      pool_size=2,
                                      strides=1,
                                      padding='SAME')

    # [1], section 3.1 CBHG Module:
    # "We further pass the processed sequence to a few fixed-width 1-D convolutions, whose outputs
    # are added with the original input sequence via residual connections [...]."

    # network.shape => (B, T, projections[-1].proj_filters)
    with tf.variable_scope('projections'):
        for i, (proj_filters, proj_kernel_size, proj_activation) in enumerate(projections):
            proj_scope = '{}-conv-{}-{}'.format(i + 1, proj_kernel_size, proj_filters)

            # network.shape => (B, T, proj_filters)
            network = conv_1d_projection(inputs=network,
                                         n_filters=proj_filters,
                                         kernel_size=proj_kernel_size,
                                         activation=proj_activation,
                                         scope=proj_scope,
                                         training=training)

    # Residual connection.
    # network.shape => (B, T, inputs.shape[-1])
    network = tf.add(network, inputs)

    # Highway network dimensionality lifter.
    # network.shape => (B, T, n_highway_units)
    network = tf.layers.dense(inputs=network,
                              units=n_highway_units,
                              activation=tf.nn.relu,
                              kernel_initializer=tf.glorot_normal_initializer(),
                              bias_initializer=tf.glorot_normal_initializer(),
                              name='lifter')

    # Multi layer highway network.
    # network.shape => (B, T, n_highway_units)
    network = highway_network(inputs=network,
                              units=n_highway_units,
                              layers=n_highway_layers,
                              scope='highway_network')

    cell_forward = tf.nn.rnn_cell.GRUCell(num_units=n_gru_units, name='gru_cell_fw')
    cell_backward = tf.nn.rnn_cell.GRUCell(num_units=n_gru_units, name='gru_cell_bw')

    # TODO: Calculate the sequence lengths that should be used.
    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell_forward,
        cell_bw=cell_backward,
        inputs=network,
        dtype=tf.float32,
        scope='gru'
    )

    # network.shape => (B, T, n_gru_units * 2)
    network = tf.concat(outputs, -1)

    return network, output_states
