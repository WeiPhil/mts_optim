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

"""Defines a displacement map variable."""

from __future__ import annotations

from collections.abc import Callable

import drjit as dr  # type: ignore
import gin
import mitsuba as mi  # type: ignore

from mts_optim.variables import variable


@gin.configurable
class DisplacementMapVariable(variable.Variable):
    """Models a displacement map variable as a wrapper of another variable.

    This currently assumes that the mesh vertices themselves are fixed.
    """

    def __init__(
        self,
        key: str,
        nested_variable: variable.Variable | Callable[..., variable.Variable],
        displacement_scale: float = 1.0,
        cubic_interpolation: bool = True,
        **kwargs,
    ):
        super().__init__(key=key, **kwargs)

        # The nested variable might need to be constructed. In that case,
        # pass down any unused keyword arguments we might want there.
        if callable(nested_variable):
            nested_variable = nested_variable(**kwargs)

        self.variable = nested_variable
        self.displacement_scale = displacement_scale
        self.cubic_interpolation = cubic_interpolation
        self.mesh_vertices = None
        self.mesh_normals = None
        self.mesh_texcoords = None
        self.register_nested_variables(nested_variable)

    def initialize(
        self,
        optimizer: mi.ad.Optimizer,
        parameters: mi.python.util.SceneParameters,
    ):
        self._fetch_mesh_data(parameters)
        super().initialize(optimizer, parameters)

    def scene_update(
        self,
        optimizer: mi.ad.Optimizer,
        parameters: mi.python.util.SceneParameters,
    ):
        self._fetch_mesh_data(parameters)
        super().scene_update(optimizer, parameters)

    def _fetch_mesh_data(self, parameters: mi.python.util.SceneParameters):
        # Return early if we already obtained this data from the scene.
        if self.mesh_vertices is not None:
            return

        # Retain a copy of the original mesh vertices and normals.
        self.mesh_vertices = dr.unravel(
            mi.Point3f, mi.Float(parameters[self.key]))
        self.mesh_normals = dr.unravel(
            mi.Vector3f,
            mi.Float(
                parameters[self.key.replace(
                    '.vertex_positions', '.vertex_normals')]
            ),
        )
        self.mesh_texcoords = dr.unravel(
            mi.Point2f,
            mi.Float(
                parameters[
                    self.key.replace('.vertex_positions', '.vertex_texcoords')
                ]
            ),
        )
        # Make sure these are not connected to any AD graph.
        self.mesh_vertices = dr.detach(self.mesh_vertices)
        self.mesh_normals = dr.detach(self.mesh_normals)
        self.mesh_texcoords = dr.detach(self.mesh_texcoords)
        dr.eval(self.mesh_vertices, self.mesh_normals, self.mesh_texcoords)

    def get_value(self, optimizer: mi.ad.Optimizer) -> mi.Float:
        # Applies the displacement map to the mesh's vertex positions.
        assert self.mesh_vertices is not None
        assert self.mesh_normals is not None
        assert self.mesh_texcoords is not None

        texture = self.variable.get_value(optimizer)
        dr.eval(texture)

        if self.cubic_interpolation:
            displacement = mi.Texture2f(
                texture).eval_cubic(self.mesh_texcoords)[0]
        else:
            displacement = mi.Texture2f(texture).eval(self.mesh_texcoords)[0]
        displacement *= self.displacement_scale
        vertex_positions = self.mesh_vertices + self.mesh_normals * displacement
        dr.eval(vertex_positions)
        assert dr.grad_enabled(vertex_positions) == dr.grad_enabled(texture)
        return dr.ravel(vertex_positions)
