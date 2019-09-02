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


def dice_coe(output, target, axis=(1, 2, 3), smooth=1e-12):
    """Soft Sørensen–Dice coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.

    Examples
    ---------
    '>>> outputs = tl.act.pixel_wise_softmax(network.outputs)'
    '>>> dice_loss = 1 - dice_coe(outputs, y_)'

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """

    assert output.dtype in [tf.float32, tf.float64]
    assert target.dtype in [tf.float32, tf.float64]

    intersection = tf.reduce_sum(output * target, axis=axis)

    a = tf.reduce_sum(output, axis=axis)
    b = tf.reduce_sum(target, axis=axis)

    score = (2. * intersection + smooth) / (a + b + smooth)
    score = tf.reduce_mean(score, name='dice_coe')
    return score
