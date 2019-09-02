#  Copyright 2019 Gabriele Valvano
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import tensorflow as tf
from tensorflow import layers

# He initializer for the layers with ReLU activation function:
he_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
b_init = tf.zeros_initializer()


class SmallUnet(object):

    def __init__(self, incoming, n_out, is_training, n_filters=16, name='Small-U-Net_2D'):
        """
        Class for UNet architecture but with fewer filters and downsampling blocks. It employs bi-dimensional
        convolution and strides. This implementation also uses batch normalization after each convolutional layer.
        :param incoming: (tensor) incoming tensor
        :param n_out: (int) number of channels for the network output. For instance, to predict a binary mask you must
                        use n_out=2 (one-hot encoding); to predict a grayscale image you must use n_out=1
        :param is_training: (tf.placeholder(dtype=tf.bool) or bool) variable to define training or test mode; it is
                        needed for the behaviour of dropout, batch normalization, ecc. (which behave differently
                        at train and test time)
        :param n_filters: (int) number of filters in the first layer. Default=16 (3x fewer than in the vanilla UNet)
        :param name: (string) variable scope for the model

        - - - - - - - - - - - - - - - -
        Notice that:
          - this implementation works for incoming tensors with shape [None, N, M, K], where N and M must be divisible
            by 8 without any rest (in fact, there are 3 pooling layers with kernels 2x2 --> input reduced to:
            [None, N/8, M/8, K'])
          - the output of the network has activation linear
        - - - - - - - - - - - - - - - -

        Example of usage:

            # build the entire unet model:
            unet = SmallUnet(incoming, n_out, is_training).build()

            # build the unet with access to the internal code:
            unet = SmallUnet(incoming, n_out, is_training)
            encoder = unet.build_encoder()
            code = unet.build_code(encoder)
            decoder = unet.build_decoder(code)
            output = unet.build_output(decoder)

        """

        # check for compatible input dimensions
        shape = incoming.get_shape().as_list()
        assert not shape[1] % 8
        assert not shape[2] % 8

        self.incoming = incoming
        self.n_out = n_out
        self.is_training = is_training
        self.nf = n_filters
        self.name = name

    def build(self, reuse=tf.AUTO_REUSE):
        """
        Build the model.
        :param reuse: (bool) if True, reuse trained weights
        """
        with tf.variable_scope(self.name, reuse=reuse):
            encoder = self.build_encoder()
            code = self.build_bottleneck(encoder)
            decoder = self.build_decoder(code)
            output = self.build_output(decoder)
            # final activation: linear
        return output

    def build_encoder(self):
        """ Encoder layers """
        with tf.variable_scope('Encoder'):
            en_brick_0, concat_0 = self._encode_brick(self.incoming, self.nf, self.is_training, scope='encode_brick_0')
            en_brick_1, concat_1 = self._encode_brick(en_brick_0, 2 * self.nf, self.is_training, scope='encode_brick_1')
            en_brick_2, concat_2 = self._encode_brick(en_brick_1, 4 * self.nf, self.is_training, scope='encode_brick_2')

        return en_brick_0, concat_0, en_brick_1, concat_1, en_brick_2, concat_2

    def build_bottleneck(self, encoder):
        """ Central layers """
        en_brick_0, concat_0, en_brick_1, concat_1, en_brick_2, concat_2 = encoder

        with tf.variable_scope('Bottleneck'):
            code = self._bottleneck_brick(en_brick_2, 8 * self.nf, self.is_training, scope='code')

        return en_brick_0, concat_0, en_brick_1, concat_1, en_brick_2, concat_2, code

    def build_decoder(self, code):
        """ Decoder layers """
        en_brick_0, concat_0, en_brick_1, concat_1, en_brick_2, concat_2, code = code

        with tf.variable_scope('Decoder'):
            dec_brick_0 = self._decode_brick(code, 4 * self.nf, self.is_training, scope='decode_brick_0')
            dec_brick_1 = self._decode_brick(dec_brick_0, 2 * self.nf, self.is_training, scope='decode_brick_1')
            dec_brick_2 = self._decode_brick(dec_brick_1, self.nf, self.is_training, scope='decode_brick_2')

        return dec_brick_2

    def build_output(self, decoder):
        """ Output layers """
        # output linear
        return self._output_layer(decoder, n_channels_out=self.n_out, scope='output')

    @staticmethod
    def _encode_brick(incoming, nb_filters, is_training, scope, trainable=True):
        """ Encoding brick: conv --> conv --> max pool.
        """
        with tf.variable_scope(scope):
            conv1 = layers.conv2d(incoming, filters=nb_filters, kernel_size=3, strides=1, padding='same',
                                  kernel_initializer=he_init, bias_initializer=b_init, trainable=trainable)
            conv1_bn = layers.batch_normalization(conv1, training=is_training, trainable=trainable)
            conv1_act = tf.nn.relu(conv1_bn)

            conv2 = layers.conv2d(conv1_act, filters=nb_filters, kernel_size=3, strides=1, padding='same',
                                  kernel_initializer=he_init, bias_initializer=b_init, trainable=trainable)
            conv2_bn = layers.batch_normalization(conv2, training=is_training, trainable=trainable)
            conv2_act = tf.nn.relu(conv2_bn)

            pool = layers.max_pooling2d(conv2_act, pool_size=2, strides=2, padding='same')

            with tf.variable_scope('concat_layer_out'):
                concat_layer_out = conv2_bn

        return pool, concat_layer_out

    @staticmethod
    def _decode_brick(incoming, nb_filters, is_training, scope):
        """ Decoding brick: deconv (up-pool) --> conv --> conv.
        """
        with tf.variable_scope(scope):
            _, old_height, old_width, __ = incoming.get_shape()
            new_height, new_width = 2.0 * old_height, 2.0 * old_width
            upsampled = tf.image.resize_nearest_neighbor(incoming, size=[new_height, new_width])
            conv1t = layers.conv2d(upsampled, filters=nb_filters, kernel_size=3, strides=1, padding='same',
                                   kernel_initializer=he_init, bias_initializer=b_init)
            conv1t_bn = layers.batch_normalization(conv1t, training=is_training)
            conv1t_act = tf.nn.relu(conv1t_bn)

            conv2 = layers.conv2d(conv1t_act, filters=nb_filters, kernel_size=3, strides=1, padding='same',
                                  kernel_initializer=he_init, bias_initializer=b_init)
            conv2_bn = layers.batch_normalization(conv2, training=is_training)
            conv2_act = tf.nn.relu(conv2_bn)

            conv3 = layers.conv2d(conv2_act, filters=nb_filters, kernel_size=3, strides=1, padding='same',
                                  kernel_initializer=he_init, bias_initializer=b_init)
            conv3_bn = layers.batch_normalization(conv3, training=is_training)
            conv3_act = tf.nn.relu(conv3_bn)

        return conv3_act

    @staticmethod
    def _bottleneck_brick(incoming, nb_filters, is_training, scope, trainable=True):
        """ Code brick: conv --> conv .
        """
        with tf.variable_scope(scope):
            code1 = layers.conv2d(incoming, filters=nb_filters, kernel_size=1, strides=1, padding='same',
                                  kernel_initializer=he_init, bias_initializer=b_init)
            code1_bn = layers.batch_normalization(code1, training=is_training, trainable=trainable)
            code1_act = tf.nn.relu(code1_bn)

            code2 = layers.conv2d(code1_act, filters=nb_filters, kernel_size=1, strides=1, padding='same',
                                  kernel_initializer=he_init, bias_initializer=b_init)
            code2_bn = layers.batch_normalization(code2, training=is_training, trainable=trainable)
            code2_act = tf.nn.relu(code2_bn)

        return code2_act

    @staticmethod
    def _output_layer(incoming, n_channels_out, scope):
        """ Output layer: conv .
        """
        with tf.variable_scope(scope):
            output = layers.conv2d(incoming, filters=n_channels_out, kernel_size=1, strides=1, padding='same',
                                   kernel_initializer=he_init, bias_initializer=b_init)
            # activation = linear
        return output
