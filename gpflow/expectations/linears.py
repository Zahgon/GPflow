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

from typing import Type, Union

import tensorflow as tf
from check_shapes import check_shapes

from .. import kernels
from .. import mean_functions as mfn
from ..inducing_variables import InducingPoints
from ..probability_distributions import DiagonalGaussian, Gaussian, MarkovGaussian
from . import dispatch
from .expectations import expectation

NoneType: Type[None] = type(None)


@dispatch.expectation.register(Gaussian, kernels.Linear, NoneType, NoneType, NoneType)
@check_shapes(
    "p: [N, D]",
    "return: [N]",
)
def _expectation_gaussian_linear(
    p: Gaussian, kernel: kernels.Linear, _: None, __: None, ___: None, nghp: None = None
) -> tf.Tensor:
    """
    Compute the expectation:
    <diag(K_{X, X})>_p(X)
        - K_{.,.} :: Linear kernel

    :return: N
    """
    pass


@dispatch.expectation.register(Gaussian, kernels.Linear, InducingPoints, NoneType, NoneType)
@check_shapes(
    "p: [N, D]",
    "inducing_variable: [M, D, P]",
    "return: [N, M]",
)
def _expectation_gaussian_linear_inducingpoints(
    p: Gaussian,
    kernel: kernels.Linear,
    inducing_variable: InducingPoints,
    _: None,
    __: None,
    nghp: None = None,
) -> tf.Tensor:
    """
    Compute the expectation:
    <K_{X, Z}>_p(X)
        - K_{.,.} :: Linear kernel

    :return: NxM
    """
    pass


@dispatch.expectation.register(Gaussian, kernels.Linear, InducingPoints, mfn.Identity, NoneType)
@check_shapes(
    "p: [N, D]",
    "inducing_variable: [M, D, P]",
    "return: [N, M, D]",
)
def _expectation_gaussian_linear_inducingpoints__identity(
    p: Gaussian,
    kernel: kernels.Linear,
    inducing_variable: InducingPoints,
    mean: mfn.Identity,
    _: None,
    nghp: None = None,
) -> tf.Tensor:
    """
    Compute the expectation:
    expectation[n] = <K_{Z, x_n} x_n^T>_p(x_n)
        - K_{.,.} :: Linear kernel

    :return: NxMxD
    """
    pass


@dispatch.expectation.register(
    MarkovGaussian, kernels.Linear, InducingPoints, mfn.Identity, NoneType
)
@check_shapes(
    "p: [N, D]",
    "inducing_variable: [M, D, P]",
    "return: [N, M, D]",
)
def _expectation_markov_linear_inducingpoints__identity(
    p: MarkovGaussian,
    kernel: kernels.Linear,
    inducing_variable: InducingPoints,
    mean: mfn.Identity,
    _: None,
    nghp: None = None,
) -> tf.Tensor:
    """
    Compute the expectation:
    expectation[n] = <K_{Z, x_n} x_{n+1}^T>_p(x_{n:n+1})
        - K_{.,.} :: Linear kernel
        - p       :: MarkovGaussian distribution (p.cov 2x(N+1)xDxD)

    :return: NxMxD
    """
    pass


@dispatch.expectation.register(
    (Gaussian, DiagonalGaussian), kernels.Linear, InducingPoints, kernels.Linear, InducingPoints
)
@check_shapes(
    "p: [N, D]",
    "feat1: [M, D, P]",
    "feat2: [M, D, P]",
    "return: [N, M, M]",
)
def _expectation_gaussian_linear_inducingpoints__linear_inducingpoints(
    p: Union[Gaussian, DiagonalGaussian],
    kern1: kernels.Linear,
    feat1: InducingPoints,
    kern2: kernels.Linear,
    feat2: InducingPoints,
    nghp: None = None,
) -> tf.Tensor:
    """
    Compute the expectation:
    expectation[n] = <Ka_{Z1, x_n} Kb_{x_n, Z2}>_p(x_n)
        - Ka_{.,.}, Kb_{.,.} :: Linear kernels
    Ka and Kb as well as Z1 and Z2 can differ from each other, but this is supported
    only if the Gaussian p is Diagonal (p.cov NxD) and Ka, Kb have disjoint active_dims
    in which case the joint expectations simplify into a product of expectations

    :return: NxMxM
    """
    pass
