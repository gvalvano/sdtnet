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
import numpy as np
from tensorflow.contrib.image import interpolate_spline
from six.moves import reduce
from architectures.mlp_in import MLP

bilinear_interpolation = tf.contrib.resampler.resampler

# He initializer for the layers with ReLU activation function:
he_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
b_init = tf.zeros_initializer()


class ThinPlateSpline2D(tf.keras.layers.Layer):

    def __init__(self, input_volume_shape, cp_dims, num_channels, inverse=False, order=2, **kwargs):

        self.vol_shape = input_volume_shape
        self.data_dimensionality = len(input_volume_shape)
        self.cp_dims = cp_dims
        self.num_channels = num_channels
        self.initial_cp_grid = None
        self.flt_grid = None
        self.inverse = inverse
        self.order = order
        super(ThinPlateSpline2D, self).__init__(**kwargs)

    def build(self, input_shape):

        self.initial_cp_grid = nDgrid(self.cp_dims)
        self.flt_grid = nDgrid(self.vol_shape)

        super(ThinPlateSpline2D, self).build(input_shape)

    # @tf.contrib.eager.defun
    def interpolate_spline_batch(self, cp_offsets_single_batch):

        warped_cp_grid = self.initial_cp_grid + cp_offsets_single_batch

        if self.inverse:
            interpolated_sample_locations = interpolate_spline(train_points=warped_cp_grid,
                                                               train_values=self.initial_cp_grid,
                                                               query_points=self.flt_grid,
                                                               order=self.order)
        else:
            interpolated_sample_locations = interpolate_spline(train_points=self.initial_cp_grid,
                                                               train_values=warped_cp_grid,
                                                               query_points=self.flt_grid,
                                                               order=self.order)

        return interpolated_sample_locations

    def call(self, args, **kwargs):

        vol, cp_offsets = args

        # rescale offsets to the volume's size:
        # cp_offsets *= 40.

        interpolated_sample_locations = tf.map_fn(self.interpolate_spline_batch, cp_offsets)[:, 0]

        interpolated_sample_locations = tf.reverse(interpolated_sample_locations, axis=[-1])

        interpolated_sample_locations = tf.multiply(interpolated_sample_locations,
                                                    [self.vol_shape[1] - 1, self.vol_shape[0] - 1])
        warped_volume = bilinear_interpolation(vol, interpolated_sample_locations)
        warped_volume = tf.reshape(warped_volume, (-1,) + tuple(self.vol_shape) + (self.num_channels,))
        return warped_volume


def nDgrid(dims, normalise=True, center=False, dtype='float32'):
    """
    returns the co-ordinates for an n-dimensional grid as a (num-points, n) shaped array
    e.g. dims=[3,3] would return:
      [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]
    if normalized == False, or:
      [[0,0],[0,0.5],[0,1.],[0.5,0],[0.5,0.5],[0.5,1.],[1.,0],[1.,0.5],[1.,1.]]
    if normalized == True.
    """
    if len(dims) == 2:
        grid = np.expand_dims(np.mgrid[:dims[0], :dims[1]].reshape((2, -1)).T, 0)
    elif len(dims) == 3:
        grid = np.expand_dims(np.mgrid[:dims[0], :dims[1], :dims[2]].reshape((3, -1)).T, 0)
    else:
        # non supported input dimension: input must be 2D image or 3D volume
        raise Exception

    if normalise:
        grid = grid / (1. * (np.array([[dims]]) - 1))

        if center:
            grid = (grid - 1) * 2

    return tf.cast(grid, dtype=dtype)


def regress_theta(input_tensor, input_times, delta_times, n_points, is_training):

    nf = 16  # num of filters

    with tf.variable_scope('Encoder'):
        en_brick_0 = _encode_brick(input_tensor, nf, is_training, scope='encode_brick_0')
        en_brick_1 = _encode_brick(en_brick_0, 2 * nf, is_training, scope='encode_brick_1')
        en_brick_2 = _encode_brick(en_brick_1, 4 * nf, is_training, scope='encode_brick_2')

    # get latent code:
    latent_code = layers.conv2d(en_brick_2, filters=8 * nf, kernel_size=3, strides=2, padding='valid')
    latent_code = tf.nn.sigmoid(latent_code)
    latent_code = tf.layers.flatten(latent_code)

    # - - - - - - -
    # condition with encoded time information (output of MLP #1) :
    latent_shape = latent_code.get_shape().as_list()
    n_fraction = 4
    n_out = reduce(lambda x, y: x * y, latent_shape[1:]) // n_fraction

    times = tf.concat((tf.expand_dims(input_times, 1),
                       tf.expand_dims(delta_times, 1)), axis=1)
    time_code = MLP(times, 128, 128, n_out, is_training, k_prob=0.8, name='MLP_in').build()
    time_code = tf.nn.tanh(time_code)

    latent_code_with_time = tf.concat((latent_code, time_code), axis=-1)

    # -------------
    # Extract the points
    # these are 2D points: hence the output are n_points pairs (x, y) --> units=2*n_points

    theta = layers.dense(latent_code_with_time, units=2*n_points, kernel_initializer='zeros', bias_initializer='zeros')
    theta = tf.reshape(theta, (-1, n_points, 2))

    return theta


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

    return pool
