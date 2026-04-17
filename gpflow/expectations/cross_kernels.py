# Copyright 2017-2020 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union, cast

import tensorflow as tf
from check_shapes import check_shapes

from .. import kernels
from ..base import TensorType
from ..inducing_variables import InducingPoints
from ..probability_distributions import DiagonalGaussian, Gaussian
from . import dispatch
from .expectations import expectation


@dispatch.expectation.register(
    (Gaussian, DiagonalGaussian),
    kernels.SquaredExponential,
    InducingPoints,
    kernels.Linear,
    InducingPoints,
)
@check_shapes(
    "p: [N, D]",
    "feat1: [M1, D, P]",
    "feat2: [M2, D, P]",
    "return: [N, M1, M2]",
)
def _expectation_gaussian_sqe_inducingpoints__linear_inducingpoints(
    p: Union[Gaussian, DiagonalGaussian],
    sqexp_kern: kernels.SquaredExponential,
    feat1: InducingPoints,
    lin_kern: kernels.Linear,
    feat2: InducingPoints,
    nghp: None = None,
) -> tf.Tensor:
    """
    Compute the expectation:
    expectation[n] = <Ka_{Z1, x_n} Kb_{x_n, Z2}>_p(x_n)
        - K_lin_{.,.} :: SqExp kernel
        - K_sqexp_{.,.} :: Linear kernel
    Different Z1 and Z2 are handled if p is diagonal and K_lin and K_sqexp have disjoint
    active_dims, in which case the joint expectations simplify into a product of expectations

    :return: NxM1xM2
    """
    pass


@dispatch.expectation.register(
    (Gaussian, DiagonalGaussian),
    kernels.Linear,
    InducingPoints,
    kernels.SquaredExponential,
    InducingPoints,
)
@check_shapes(
    "p: [N, D]",
    "feat1: [M1, D, P]",
    "feat2: [M2, D, P]",
    "return: [N, M1, M2]",
)
def _expectation_gaussian_linear_inducingpoints__sqe_inducingpoints(
    p: Union[Gaussian, DiagonalGaussian],
    lin_kern: kernels.Linear,
    feat1: InducingPoints,
    sqexp_kern: kernels.SquaredExponential,
    feat2: InducingPoints,
    nghp: None = None,
) -> tf.Tensor:
    """
    Compute the expectation:
    expectation[n] = <Ka_{Z1, x_n} Kb_{x_n, Z2}>_p(x_n)
        - K_lin_{.,.} :: Linear kernel
        - K_sqexp_{.,.} :: sqexp kernel
    Different Z1 and Z2 are handled if p is diagonal and K_lin and K_sqexp have disjoint
    active_dims, in which case the joint expectations simplify into a product of expectations

    :return: NxM1xM2
    """
    pass
