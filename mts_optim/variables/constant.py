# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements a constant "no-op" Variable class.

This is convenient to disable optimization of a given attribute without having
to change much other code.
"""

from __future__ import annotations

import drjit as dr  # type: ignore
import mitsuba as mi  # type: ignore

from mts_optim.variables import array


class Constant(array.ArrayVariable):
    """Constant optimization Variable.

    This is a Variable that is not optimized. This is useful as a fallback
    to disable optimization of some Variables in code that assumes the Variable
    interface.
    """

    def __init__(
        self,
        key: str,
        **kwargs,
    ):
        super().__init__(
            key=key,
            **kwargs,
        )

    def get_value(self, optimizer: mi.ad.Optimizer) -> mi.TensorXf:
        # We make the parameter constant by never propagating gradients to it.
        return dr.detach(super().get_value(optimizer))
