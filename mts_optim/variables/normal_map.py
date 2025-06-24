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

"""Defines a tangent space normal map variable."""

from __future__ import annotations

from collections.abc import Callable

import drjit as dr  # type: ignore
import gin
import mitsuba as mi  # type: ignore

from mts_optim.variables import variable


@gin.configurable
class NormalMapVariable(variable.Variable):
    """Models a normal map as the derivatives of an inner bump map.

    This is similar to directly optimizing a bump map, but instead of
    using analytic texture space derivatives (as in Mitsuba's `bumpmap` plugin),
    this uses finite differences derivatives to estimate normals from an
    underlying bump map. The resulting normals are then interpolated using
    bilinear interpolation, which results in a smooth appearance compared to
    directly optimizing a bump map. The resulting normal field is integrable,
    since it corresponds to the surface described by the underlying height
    map representation.

    Attributes:
      nested_variable: A nested variable that represents a height/bump map.
      bump_scale: A scale factor to be applied when converting this bump map to a
        normal map. Lowering that scale factor reduces the sensitivity of the
        tangent space normal to changes in the bump map.
    """

    def __init__(
        self,
        key: str,
        nested_variable: variable.Variable | Callable[..., variable.Variable],
        bump_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(key=key, **kwargs)

        # The nested variable might need to be constructed. In that case,
        # pass down any unused keyword arguments we might want there.
        if callable(nested_variable):
            nested_variable = nested_variable(**kwargs)

        self.bump_scale = bump_scale
        self.nested_variable = nested_variable
        self.register_nested_variables(self.nested_variable)

    def get_value(self, optimizer: mi.ad.Optimizer) -> mi.TensorXf:
        bump = self.nested_variable.get_value(optimizer)

        # Compute normal map using discrete finite differences on bump map.
        shape = bump.shape
        assert len(shape) == 3 and shape[-1] == 1
        y, x = dr.meshgrid(
            dr.arange(mi.Int32, shape[0]),
            dr.arange(mi.Int32, shape[1]),
            indexing='ij',
        )
        x_neighbor = dr.minimum(x + 1, shape[1] - 1)
        y_neighbor = dr.minimum(y + 1, shape[0] - 1)
        x_value = dr.gather(mi.Float, bump.array, y * shape[1] + x_neighbor)
        y_value = dr.gather(mi.Float, bump.array, y_neighbor * shape[1] + x)
        dx = (x_value - bump.array) * shape[1] * self.bump_scale
        dy = (y_value - bump.array) * shape[0] * self.bump_scale
        normal = dr.normalize(mi.Vector3f(dx, dy, 1.0))
        # Map to range [0, 1] as needed by Mitsuba.
        normal = 0.5 * (normal + 1.0)
        return mi.TensorXf(dr.ravel(normal), shape=(shape[0], shape[1], 3))
