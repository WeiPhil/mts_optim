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

"""Defines a look-at transformation variable."""

from __future__ import annotations

import mitsuba as mi  # type: ignore

from mts_optim.variables import array
from mts_optim.variables import constant
from mts_optim.variables import variable


class LookAtTransformVariable(variable.Variable):
    """Represents a look-at transformation.

    This variable allows to optimize a transformation using an origin and target
    point that describe a look-at transformation.
    """

    def __init__(
        self,
        key: str,
        *,
        origin: mi.Point3f,
        target: mi.Point3f,
        up: mi.Vector3f | None = None,
        local_transform: mi.Transform4f | float = 1.0,
        optimize_origin: bool = True,
        is_scene_parameter: bool = True,
        **kwargs,
    ):
        """Initializes a LookAtTransformVariable.

        Args:
          key: the variable's string key.
          origin: the origin point for the lookat transformation.
          target: the target point for the lookat transformation.
          up: the lookat transformation's up vector, which is assumed to be fixed
            (i.e., it is not optimized).
          local_transform: an optional transformation that is applied before the
            lookat transformation (e.g., a scaling transform). If a float value is
            provided, it is interpreted as a uniform scaling.
          optimize_origin: if set to True, optimizes the origin of the
            lookat-transform. If set to False, only the target point will be
            optimized (default: True).
          is_scene_parameter: Whether this variable corresponds to a Mitsuba scene
            parameter.
          **kwargs: Remaining keyword arguments.
        """

        super().__init__(
            key=key,
            **kwargs,
        )
        key = self.key + '_origin'
        if optimize_origin:
            self._origin = array.ArrayVariable(
                key,
                initial_value=origin,
                is_scene_parameter=False,
                **kwargs,
            )
        else:
            self._origin = constant.Constant(key, initial_value=origin)

        key = self.key + '_target'
        self._target = array.ArrayVariable(
            key,
            initial_value=target,
            is_scene_parameter=False,
            **kwargs,
        )
        if up is None:
            up = mi.Vector3f(0, 1, 0)
        self._up = up
        if isinstance(local_transform, float):
            local_transform = mi.Transform4f().scale(local_transform)
        self._local_transform = local_transform

        self.register_nested_variables(self._origin, self._target)

    def get_value(self, optimizer: mi.ad.Optimizer) -> mi.Transform4f:
        to_world = mi.Transform4f().look_at(
            self._origin.get_value(optimizer),
            self._target.get_value(optimizer),
            self._up,
        )
        return to_world @ self._local_transform
