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

from typing import Any, Callable, Optional, Type, Union, cast

import numpy as np
import tensorflow as tf
from check_shapes import check_shapes

from .. import kernels
from .. import mean_functions as mfn
from ..base import TensorType
from ..covariances import Kuf
from ..inducing_variables import InducingVariables
from ..probability_distributions import DiagonalGaussian, Gaussian, MarkovGaussian
from ..quadrature import mvnquad
from . import dispatch
from .expectations import ExpectationObject, PackedExpectationObject, quadrature_expectation

register = dispatch.quadrature_expectation.register


NoneType: Type[None] = type(None)
EllipsisType = Any


def get_eval_func(
    obj: ExpectationObject,
    inducing_variable: Optional[InducingVariables],
    slice: Union[slice, EllipsisType, None] = None,
) -> Callable[[TensorType], tf.Tensor]:
    """
    Return the function of interest (kernel or mean) for the expectation
    depending on the type of :obj: and whether any inducing are given
    """

    slice = ... if slice is None else slice
    if inducing_variable is not None:
        # kernel + inducing_variable combination
        if not isinstance(inducing_variable, InducingVariables) or not isinstance(
            obj, kernels.Kernel
        ):
            raise TypeError("If `inducing_variable` is supplied, `obj` must be a kernel.")
        return lambda x: tf.transpose(Kuf(inducing_variable, obj, x))[slice]
    elif isinstance(obj, mfn.MeanFunction):
        return lambda x: obj(x)[slice]  # type: ignore[misc]
    elif isinstance(obj, kernels.Kernel):
        return lambda x: obj(x, full_cov=False)  # type: ignore[call-arg, misc]

    raise NotImplementedError()


@dispatch.quadrature_expectation.register(
    (Gaussian, DiagonalGaussian),
    object,
    (InducingVariables, NoneType),
    object,
    (InducingVariables, NoneType),
)
@check_shapes(
    "p: [N, D]",
    "inducing_variable1: [M1, D, P]",
    "inducing_variable2: [M2, D, P]",
    "return: [N, ...]",
)
def _quadrature_expectation_gaussian(
    p: Union[Gaussian, DiagonalGaussian],
    obj1: ExpectationObject,
    inducing_variable1: Optional[InducingVariables],
    obj2: ExpectationObject,
    inducing_variable2: Optional[InducingVariables],
    nghp: Optional[int] = None,
) -> tf.Tensor:
    """
    General handling of quadrature expectations for Gaussians and DiagonalGaussians
    Fallback method for missing analytic expectations
    """
    pass


@dispatch.quadrature_expectation.register(
    MarkovGaussian, object, (InducingVariables, NoneType), object, (InducingVariables, NoneType)
)
@check_shapes(
    "p: [N, D]",
    "inducing_variable1: [M1, D, P]",
    "inducing_variable2: [M2, D, P]",
    "return: [N, ...]",
)
def _quadrature_expectation_markov(
    p: MarkovGaussian,
    obj1: ExpectationObject,
    inducing_variable1: Optional[InducingVariables],
    obj2: ExpectationObject,
    inducing_variable2: Optional[InducingVariables],
    nghp: Optional[int] = None,
) -> tf.Tensor:
    """
    Handling of quadrature expectations for Markov Gaussians (useful for time series)
    Fallback method for missing analytic expectations wrt Markov Gaussians
    Nota Bene: obj1 is always associated with x_n, whereas obj2 always with x_{n+1}
               if one requires e.g. <x_{n+1} K_{x_n, Z}>_p(x_{n:n+1}), compute the
               transpose and then transpose the result of the expectation
    """
    pass
