import tensorflow as tf


class BasicBlock(tf.keras.layers.Layer):
    """ Implementation of a basic block layer used for the ResNet type I model. """

    def __init__(self, filter_num, stride=1, l2_amount=0.1):
        """
        Initialises the basic block by providing the number of filter, strides and l2 regularisation amount.

        :param filter_num: the number of filters in both of the two dimensional convolution layers.
        :param stride: the stride for the two dimensional convolution layers.
        :param l2_amount: the l2 regularisation amount of the convolution layers.
        """
        super(BasicBlock, self).__init__()
        self.l2_regularization = tf.keras.regularizers.l2(l2_amount)
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same",
                                            bias_regularizer=self.l2_regularization,
                                            kernel_regularizer=self.l2_regularization)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same",
                                            bias_regularizer=self.l2_regularization,
                                            kernel_regularizer=self.l2_regularization)
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                       kernel_size=(1, 1),
                                                       strides=stride,
                                                       bias_regularizer=self.l2_regularization,
                                                       kernel_regularizer=self.l2_regularization))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    @tf.function
    def call(self, inputs, training=None, **kwargs):
        """
        Calls the forward pass through the basic block layer.

        :param inputs: the inputs to the layer.
        :param training: if the layers input is from the training process or not.
        """
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output


class BottleNeck(tf.keras.layers.Layer):
    """ Implementation of a bottle neck layer used for the ResNet type II model. """

    def __init__(self, filter_num, stride=1, l2_amount=0.1):
        """
        Initialises the bottle neck layer by providing the number of filter, strides and l2 regularisation amount.

        :param filter_num: the number of filters in both of the two dimensional convolution layers.
        :param stride: the stride for the two dimensional convolution layers.
        :param l2_amount: the l2 regularisation amount of the convolution layers.
        """
        super(BottleNeck, self).__init__()
        self.l2_regularization = tf.keras.regularizers.l2(l2_amount)
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same',
                                            bias_regularizer=self.l2_regularization,
                                            kernel_regularizer=self.l2_regularization)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding='same',
                                            bias_regularizer=self.l2_regularization,
                                            kernel_regularizer=self.l2_regularization)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters=filter_num * 4,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same',
                                            bias_regularizer=self.l2_regularization,
                                            kernel_regularizer=self.l2_regularization)
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.downsample = tf.keras.Sequential()
        self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num * 4,
                                                   kernel_size=(1, 1),
                                                   strides=stride,
                                                   bias_regularizer=self.l2_regularization,
                                                   kernel_regularizer=self.l2_regularization))
        self.downsample.add(tf.keras.layers.BatchNormalization())

    @tf.function
    def call(self, inputs, training=None, **kwargs):
        """
        Calls the forward pass through the bottle neck layer.

        :param inputs: the inputs to the layer.
        :param training: if the layers input is from the training process or not.
        """
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output
