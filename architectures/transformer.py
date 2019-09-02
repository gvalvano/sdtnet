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
from architectures.small_unet import SmallUnet
from architectures.mlp_in import MLP
from six.moves import reduce
from architectures.layers.rounding_layer import rounding_layer

# He initializer for the layers with ReLU activation function:
he_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
b_init = tf.zeros_initializer()


class Transformer(object):

    def __init__(self, n_channels, is_training, n_filters=16, name='FramePredictor'):
        """
        Class for FramePredictor architecture. It predict the future frames at the future time points, with:
          future_times = input_times + delta_times

        - - - - - - - - - - - - - - - -
        Notice that:
          - this implementation works for incoming tensors with shape [None, N, M, K], where N and M must be fully
            divisible by 8 (there are 3 pooling layers with 2x2 kernels --> input reduced to: [None, N/8, M/8, K'])
          - the output of the network is binary
        - - - - - - - - - - - - - - - -

        Example of usage:

            # build the entire model:
            fp = FramePredictor(incoming, n_out, is_training).build()
            soft_output_frames = fp.get_soft_output_frames()
            hard_output_frames = fp.get_hard_output_frames()

        """
        self.n_channels = n_channels
        self.is_training = is_training
        self.nf = n_filters
        self.name = name
        self.theta_values = None
        self.soft_output_frames = None
        self.hard_output_frames = None

    def build(self, input_frames, input_times, delta_times, reuse=tf.AUTO_REUSE):
        """
        Build the model.
        :param input_frames: input frames
        :param input_times: time associated to the input frames
        :param delta_times: delta time to move in future
        :param reuse: reuse mode
        :return: prediction of the future frames at time: future_times = input_times + delta_times
        """
        with tf.variable_scope(self.name, reuse=reuse):

            # - - - - - - -
            # build only UNet encoder for now:
            unet = SmallUnet(incoming=input_frames, n_out=self.n_channels, n_filters=self.nf, is_training=self.is_training)
            encoder = unet.build_encoder()

            # get latent code:
            latent_code = encoder[-2]  # (this is the output layer of the encoder)

            # - - - - - - -
            # condition with encoded time information (output of MLP #1) :
            latent_shape = latent_code.get_shape().as_list()
            n_fraction = 4
            time_shape = [-1, latent_shape[1], latent_shape[2], (latent_shape[3] // n_fraction)]
            n_out = reduce(lambda x, y: x * y, latent_shape[1:]) // n_fraction

            times = tf.concat((tf.expand_dims(input_times, 1),
                               tf.expand_dims(delta_times, 1)), axis=1)
            mlp_in = MLP(times, 128, 128, n_out, self.is_training, k_prob=0.8, name='MLP_in').build()
            time_code = tf.reshape(mlp_in, shape=time_shape)
            
            # define activations for time and latent code, these will be accessible using:
            # [str(op.name) for op in tf.get_default_graph().get_operations() if '_code' in op.name]
            time_code = tf.nn.sigmoid(time_code, name='time_code')
            latent_code = tf.nn.sigmoid(latent_code, name='latent_code')

            latent_code_with_time = tf.concat((latent_code, time_code), axis=-1)

            # - - - - - - -
            # build the rest of the UNet
            _encoder = [el for el in encoder]
            _encoder[-2] = latent_code_with_time
            code = unet.build_bottleneck(_encoder)
            decoder = unet.build_decoder(code)
            decoded_input = tf.nn.tanh(unet.build_output(decoder))   # output in range [-1, +1]
            # decoded_input = tf.nn.sigmoid(unet.build_output(decoder))   # output in range [0, +1]

            # - - - - - - -
            # add residual connection
            self.soft_output_frames = decoded_input + input_frames

            output_frames = tf.nn.softmax(self.soft_output_frames)

            with tf.variable_scope('RoundingLayer'):
                self.hard_output_frames = rounding_layer(output_frames)

        return self

    def get_soft_output_frames(self):
        return self.soft_output_frames

    def get_hard_output_frames(self):
        return self.hard_output_frames
