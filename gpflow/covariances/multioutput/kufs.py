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

from typing import Callable, Union

import tensorflow as tf
from check_shapes import check_shapes

from ...base import TensorType
from ...inducing_variables import (
    FallbackSeparateIndependentInducingVariables,
    FallbackSharedIndependentInducingVariables,
    InducingPoints,
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)
from ...kernels import (
    LinearCoregionalization,
    MultioutputKernel,
    SeparateIndependent,
    SharedIndependent,
)
from ..dispatch import Kuf


@Kuf.register(InducingPoints, MultioutputKernel, object)
@check_shapes(
    "inducing_variable: [M, D, 1]",
    "Xnew: [batch..., N, D]",
    "return: [M, P, batch..., N, P]",
)
def Kuf_generic(
    inducing_variable: InducingPoints, kernel: MultioutputKernel, Xnew: TensorType
) -> tf.Tensor:
    pass


@Kuf.register(SharedIndependentInducingVariables, SharedIndependent, object)
@check_shapes(
    "inducing_variable: [M, D, P]",
    "Xnew: [batch..., N, D]",
    "return: [M, batch..., N]",
)
def Kuf_shared_shared(
    inducing_variable: SharedIndependentInducingVariables,
    kernel: SharedIndependent,
    Xnew: tf.Tensor,
) -> tf.Tensor:
    pass


@Kuf.register(SeparateIndependentInducingVariables, SharedIndependent, object)
@check_shapes(
    "inducing_variable: [M, D, P]",
    "Xnew: [batch..., N, D]",
    "return: [L, M, batch..., N]",
)
def Kuf_separate_shared(
    inducing_variable: SeparateIndependentInducingVariables,
    kernel: SharedIndependent,
    Xnew: TensorType,
) -> tf.Tensor:
    pass


@Kuf.register(SharedIndependentInducingVariables, SeparateIndependent, object)
@check_shapes(
    "inducing_variable: [M, D, P]",
    "Xnew: [batch..., N, D]",
    "return: [L, M, batch..., N]",
)
def Kuf_shared_separate(
    inducing_variable: SharedIndependentInducingVariables,
    kernel: SeparateIndependent,
    Xnew: TensorType,
) -> tf.Tensor:
    pass


@Kuf.register(SeparateIndependentInducingVariables, SeparateIndependent, object)
@check_shapes(
    "inducing_variable: [M, D, P]",
    "Xnew: [batch..., N, D]",
    "return: [L, M, batch..., N]",
)
def Kuf_separate_separate(
    inducing_variable: SeparateIndependentInducingVariables,
    kernel: SeparateIndependent,
    Xnew: TensorType,
) -> tf.Tensor:
    pass


@check_shapes(
    "inducing_variable: [M, D, L]",
    "Xnew: [batch..., N, D]",
    "return: [M, L, batch..., N, P]",
)
def _fallback_Kuf(
    kuf_impl: Callable[
        [
            Union[SeparateIndependentInducingVariables, SharedIndependentInducingVariables],
            LinearCoregionalization,
            TensorType,
        ],
        tf.Tensor,
    ],
    inducing_variable: Union[
        SeparateIndependentInducingVariables, SharedIndependentInducingVariables
    ],
    kernel: LinearCoregionalization,
    Xnew: TensorType,
) -> tf.Tensor:
    pass


@Kuf.register(
    FallbackSeparateIndependentInducingVariables,
    LinearCoregionalization,
    object,
)
@check_shapes(
    "inducing_variable: [M, D, L]",
    "kernel.W: [P, L]",
    "Xnew: [batch..., N, D]",
    "return: [M, L, batch..., N, P]",
)
def Kuf_fallback_separate_linear_coregionalization(
    inducing_variable: FallbackSeparateIndependentInducingVariables,
    kernel: LinearCoregionalization,
    Xnew: TensorType,
) -> tf.Tensor:
    pass


@Kuf.register(
    FallbackSharedIndependentInducingVariables,
    LinearCoregionalization,
    object,
)
@check_shapes(
    "inducing_variable: [M, D, P]",
    "kernel.W: [P, L]",
    "Xnew: [batch..., N, D]",
    "return: [M, L, batch..., N, P]",
)
def Kuf_fallback_shared_linear_coregionalization(
    inducing_variable: FallbackSharedIndependentInducingVariables,
    kernel: LinearCoregionalization,
    Xnew: TensorType,
) -> tf.Tensor:
    pass


@Kuf.register(SharedIndependentInducingVariables, LinearCoregionalization, object)
@check_shapes(
    "inducing_variable: [M, D, L]",
    "kernel.W: [P, L]",
    "Xnew: [batch..., N, D]",
    "return: [L, M, batch..., N]",
)
def Kuf_shared_linear_coregionalization(
    inducing_variable: SharedIndependentInducingVariables,
    kernel: LinearCoregionalization,
    Xnew: TensorType,
) -> tf.Tensor:
    pass


@Kuf.register(SeparateIndependentInducingVariables, LinearCoregionalization, object)
@check_shapes(
    "inducing_variable: [M, D, L]",
    "kernel.W: [P, L]",
    "Xnew: [batch..., N, D]",
    "return: [L, M, batch..., N]",
)
def Kuf_separate_linear_coregionalization(
    inducing_variable: SeparateIndependentInducingVariables,
    kernel: LinearCoregionalization,
    Xnew: TensorType,
) -> tf.Tensor:
    pass
