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

# noqa: F811

from typing import Type, Union

import tensorflow as tf
from check_shapes import check_shapes

from .. import mean_functions as mfn
from ..probability_distributions import Gaussian
from . import dispatch
from .expectations import expectation

NoneType: Type[None] = type(None)


@dispatch.expectation.register(Gaussian, (mfn.Linear, mfn.Constant), NoneType, NoneType, NoneType)
@check_shapes(
    "p: [N, D]",
    "return: [N, Q]",
)
def _expectation_gaussian_linear(
    p: Gaussian,
    mean: Union[mfn.Linear, mfn.Constant],
    _: None,
    __: None,
    ___: None,
    nghp: None = None,
) -> tf.Tensor:
    """
    Compute the expectation:
    <m(X)>_p(X)
        - m(x) :: Linear, Identity or Constant mean function

    :return: NxQ
    """
    pass


@dispatch.expectation.register(Gaussian, mfn.Constant, NoneType, mfn.Constant, NoneType)
@check_shapes(
    "p: [N, D]",
    "return: [N, Q1, Q2]",
)
def _expectation_gaussian_constant__constant(
    p: Gaussian, mean1: mfn.Constant, _: None, mean2: mfn.Constant, __: None, nghp: None = None
) -> tf.Tensor:
    """
    Compute the expectation:
    expectation[n] = <m1(x_n)^T m2(x_n)>_p(x_n)
        - m1(.), m2(.) :: Constant mean functions

    :return: NxQ1xQ2
    """
    pass


@dispatch.expectation.register(Gaussian, mfn.Constant, NoneType, mfn.MeanFunction, NoneType)
@check_shapes(
    "p: [N, D]",
    "return: [N, Q1, Q2]",
)
def _expectation_gaussian_constant__meanfunction(
    p: Gaussian, mean1: mfn.Constant, _: None, mean2: mfn.Constant, __: None, nghp: None = None
) -> tf.Tensor:
    """
    Compute the expectation:
    expectation[n] = <m1(x_n)^T m2(x_n)>_p(x_n)
        - m1(.) :: Constant mean function
        - m2(.) :: General mean function

    :return: NxQ1xQ2
    """
    pass


@dispatch.expectation.register(Gaussian, mfn.MeanFunction, NoneType, mfn.Constant, NoneType)
@check_shapes(
    "p: [N, D]",
    "return: [N, Q1, Q2]",
)
def _expectation_gaussian_meanfunction__constant(
    p: Gaussian,
    mean1: mfn.MeanFunction,
    _: None,
    mean2: mfn.MeanFunction,
    __: None,
    nghp: None = None,
) -> tf.Tensor:
    """
    Compute the expectation:
    expectation[n] = <m1(x_n)^T m2(x_n)>_p(x_n)
        - m1(.) :: General mean function
        - m2(.) :: Constant mean function

    :return: NxQ1xQ2
    """
    pass


@dispatch.expectation.register(Gaussian, mfn.Identity, NoneType, mfn.Identity, NoneType)
@check_shapes(
    "p: [N, D]",
    "return: [N, D, D]",
)
def _expectation_gaussian_identity__identity(
    p: Gaussian, mean1: mfn.Identity, _: None, mean2: mfn.Identity, __: None, nghp: None = None
) -> tf.Tensor:
    """
    Compute the expectation:
    expectation[n] = <m1(x_n)^T m2(x_n)>_p(x_n)
        - m1(.), m2(.) :: Identity mean functions

    :return: NxDxD
    """
    pass


@dispatch.expectation.register(Gaussian, mfn.Identity, NoneType, mfn.Linear, NoneType)
@check_shapes(
    "p: [N, D]",
    "return: [N, D, Q]",
)
def _expectation_gaussian_identity__linear(
    p: Gaussian, mean1: mfn.Identity, _: None, mean2: mfn.Linear, __: None, nghp: None = None
) -> tf.Tensor:
    """
    Compute the expectation:
    expectation[n] = <m1(x_n)^T m2(x_n)>_p(x_n)
        - m1(.) :: Identity mean function
        - m2(.) :: Linear mean function

    :return: NxDxQ
    """
    pass


@dispatch.expectation.register(Gaussian, mfn.Linear, NoneType, mfn.Identity, NoneType)
@check_shapes(
    "p: [N, D]",
    "return: [N, Q, D]",
)
def _expectation_gaussian_linear__identity(
    p: Gaussian, mean1: mfn.Linear, _: None, mean2: mfn.Identity, __: None, nghp: None = None
) -> tf.Tensor:
    """
    Compute the expectation:
    expectation[n] = <m1(x_n)^T m2(x_n)>_p(x_n)
        - m1(.) :: Linear mean function
        - m2(.) :: Identity mean function

    :return: NxQxD
    """
    pass


@dispatch.expectation.register(Gaussian, mfn.Linear, NoneType, mfn.Linear, NoneType)
@check_shapes(
    "p: [N, D]",
    "return: [N, Q1, Q2]",
)
def _expectation_gaussian_linear__linear(
    p: Gaussian, mean1: mfn.Linear, _: None, mean2: mfn.Linear, __: None, nghp: None = None
) -> tf.Tensor:
    """
    Compute the expectation:
    expectation[n] = <m1(x_n)^T m2(x_n)>_p(x_n)
        - m1(.), m2(.) :: Linear mean functions

    :return: NxQ1xQ2
    """
    pass
