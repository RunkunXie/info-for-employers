import tensorflow as tf

class Model(object):
    """
    Our Deep Convolutional Neural Network.
    """

    def __init__(self):
        pass

    @staticmethod
    def inference(input_x,
                  conv_featmap=[48, 64, 128, 160, 192, 192, 192, 192],
                  conv_kernel_size=[5, 5, 5, 5, 5, 5, 5, 5],
                  pooling_stride=[2, 1, 2, 1, 2, 1, 2, 1],
                  pooling_size=[2, 2, 2, 2, 2, 2, 2, 2],
                  fc_units=[3072, 3072],
                  drop_rate=0.1):
        """
        Pass input through the model, inference digit length and digit 1 to 5. The output of the 10 hidden layers are
        pass through 6 loccaly connected layers, to inference digit length and digit 1 to 5.
        Model Structure:
            8 Hidden Convolutional Layers
            2 Fully Connected Layers
            1 Locally Connected Layers

        :param input_x: input, shape=(None, 64, 64, 3)
        :param conv_featmap: number of convolution filters at each conv layer
        :param conv_kernel_size: kernel size of each conv layer
        :param pooling_stride: pooling stride of each pool layer
        :param pooling_size: pooling size of each pool layer
        :param fc_units: number of neurons in fully connected layers
        :param drop_rate: dropout rate
        :return:
            output_length: inferred digit length
            output_digits: inferred numbers, stack of [digit1, digit2, digit3, digit4, digit5]
        """

        # 8 Hidden Convolutional Layer
        hidden = input_x
        for i in range(8):
            with tf.variable_scope("conv_layers_"+str(i+1), reuse=tf.AUTO_REUSE):
                # convolution
                conv = tf.layers.conv2d(hidden, filters=conv_featmap[i], kernel_size=[conv_kernel_size[i], conv_kernel_size[i]], padding='same')

                # batch-norm
                norm = tf.layers.batch_normalization(conv)

                # activation
                activation = tf.nn.relu(norm)

                # pooling
                pool = tf.layers.max_pooling2d(activation, pool_size=[pooling_size[i], pooling_size[i]],
                                               strides=pooling_stride[i], padding='same')

                # dropout
                dropout = tf.layers.dropout(pool, rate=drop_rate)
                
                # update input for the next hidden layer
                hidden = dropout

        # 2 Fully Connected Layers
        flatten = tf.reshape(hidden, [-1, 4 * 4 * 192])
        for i in range(2):
            with tf.variable_scope("fc_layers_"+str(i+1), reuse=tf.AUTO_REUSE):
                dense = tf.layers.dense(flatten, units=fc_units[i], activation=tf.nn.relu)
                flatten = dense

        # 1 Locally Connected Layers for each output - 6 different layers
        # Locally Connected Layers for digit_length
        with tf.variable_scope('digit_length', reuse=tf.AUTO_REUSE):
            dense = tf.layers.dense(flatten, units=7)
            length = dense

        # Locally Connected Layers for digit 1 to 5
        digits = []
        for i in range(5):
            with tf.variable_scope("digit"+str(i+1), reuse=tf.AUTO_REUSE):
                dense = tf.layers.dense(flatten, units=11)
                digits.append(dense)

        # Output
        # the following line of codes are referenced from online resources
        output_length, output_digits = length, tf.stack([digit for digit in digits], axis=1)

        return output_length, output_digits

    @staticmethod
    def loss(output_length, output_digits, length_labels, digits_labels):
        """
        Compute loss for digit length and digits

        :param output_length: inferred digit length
        :param output_digits: inferred numbers, stack of [digit1, digit2, digit3, digit4, digit5]
        :param length_labels: true digit length
        :param digits_labels: true numbers
        :return:
        """

        # cross entropy for softmax output
        length_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=length_labels, logits=output_length))
        digit1_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 0], logits=output_digits[:, 0, :]))
        digit2_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 1], logits=output_digits[:, 1, :]))
        digit3_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 2], logits=output_digits[:, 2, :]))
        digit4_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 3], logits=output_digits[:, 3, :]))
        digit5_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 4], logits=output_digits[:, 4, :]))

        # total loss
        loss = length_cross_entropy + digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy + digit4_cross_entropy + digit5_cross_entropy

        return loss
