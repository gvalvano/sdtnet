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


class MLP(object):

    def __init__(self, incoming, n_in, n_hidden, n_out, is_training, k_prob=1.0, name='MLP'):
        """
        Class for 3-layered multilayer perceptron (MLP).
        :param incoming: (tensor) incoming tensor
        :param n_in: (int) number of input units
        :param n_hidden: (int) number of hidden units
        :param n_out: (int) number of output units
        :param is_training: (tf.placeholder(dtype=tf.bool) or bool) variable to define training or test mode; it is
                        needed for the behaviour of dropout (which is different at train or test time)
        :param k_prob: (float) keep probability for dropout layer. Default = 1, i.e. no dropout applied. A common value
                        for keep probability is 0.8 (e.g. 80% of active units at training time)
        :param name: (string) variable scope for the MLP
        """

        assert 0.0 <= k_prob <= 1.0
        self.incoming = incoming
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.is_training = is_training
        self.k_prob = k_prob
        self.name = name

    def build(self):
        """
        Build the MLP model.
        """
        # keep_prob = tf.cond(tf.equal(self.is_training, tf.constant(True)), lambda: self.k_prob, lambda: 1.0)

        with tf.variable_scope(self.name):
            incoming = layers.flatten(self.incoming)

            input_layer = layers.dense(incoming, self.n_in, kernel_initializer=he_init, bias_initializer=b_init)
            input_layer = tf.layers.batch_normalization(input_layer)
            input_layer = tf.nn.relu(input_layer)

            hidden_layer = layers.dense(input_layer, self.n_hidden, kernel_initializer=he_init, bias_initializer=b_init)
            hidden_layer = tf.layers.batch_normalization(hidden_layer)
            hidden_layer = tf.nn.relu(hidden_layer)

            output_layer = layers.dense(hidden_layer, self.n_out, bias_initializer=b_init)
            output_layer = tf.layers.batch_normalization(output_layer)

            # final activation: linear
        return output_layer
