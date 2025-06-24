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

"""Defines a multiresolution image pyramid variable."""

from __future__ import annotations

from collections.abc import Callable

import drjit as dr  # type: ignore
import mitsuba as mi  # type: ignore

from mts_optim.utils import losses
from mts_optim.variables import laplacian_smoothing
from mts_optim.variables import tensor
from mts_optim.variables import variable


class DifferentiableMipmapVariable(variable.Variable):
    """Represents a variable which differentably downsamples a tensor for a mipmap scene parameter.

    Attributes:
      n_levels: Number of levels of scene mipmap.
    """

    def __init__(
        self,
        key: str,
        nested_tensor: (
            tensor.TensorVariable
            | laplacian_smoothing.LaplacianSmoothingVariable
            | Callable[..., tensor.TensorVariable]
        ),
        **kwargs,
    ):
        super().__init__(key=key, **kwargs)
        # The nested variable might need to be constructed. In that case,
        # pass down any unused keyword arguments we might want there.
        if callable(nested_tensor):
            nested_tensor = nested_tensor(**kwargs)

        self.n_levels = None
        self.factor = None
        self.shape = None
        self.buffer_offsets = None
        self.storage_type = None
        self.nested_tensor = nested_tensor
        if isinstance(self.nested_tensor, tensor.TensorVariable):
            self.register_nested_variables(nested_tensor)

    def _extract_mipmap_parameters_from_scene_parameters(
        self, scene_parameters: mi.python.util.SceneParameters
    ):
        if f'{self.key}.base_mip_shape' not in scene_parameters.keys():
            raise ValueError(
                f'{self.key} does not represent a flattened mipmap in the scene'
                ' parameters!'
            )
        self.shape = tuple(scene_parameters[f'{self.key}' + '.base_mip_shape'])
        self.factor = scene_parameters[f'{self.key}' + '.mip_factor']
        self.buffer_offsets = mi.UInt32(
            scene_parameters[f'{self.key}' + '.flat_buffer_offsets']
        )
        self.n_levels = len(self.buffer_offsets) - 1
        if self.shape != self.nested_tensor.shape and (
            isinstance(self.nested_tensor, tensor.TensorVariable)
            and self.nested_tensor.upsampling_schedule is None
        ):
            raise ValueError(
                f'Shape of nested tensor {self.nested_tensor.shape} does not match'
                f' shape of highest resolution mipmap {self.shape}! This is only'
                ' allowed if the nested tensor has an upsampling schedule.'
            )

        self.storage_type = None
        if self.shape[2] == 1:
            self.storage_type = mi.Float
        else:
            assert self.shape[2] == 3
            self.storage_type = mi.Color3f

    def initialize(
        self,
        optimizer: mi.ad.Optimizer,
        parameters: mi.python.util.SceneParameters,
    ):
        if isinstance(self.nested_tensor, tensor.TensorVariable):
            super().initialize(optimizer, parameters)
        else:
            self.nested_tensor.initialize(optimizer, parameters)
        self._extract_mipmap_parameters_from_scene_parameters(parameters)

    def _get_mipmap_buffer_offsets_from_tensor(
        self, image: mi.TensorXf
    ) -> mi.UInt32:
        # Rebuild the buffer offsets for the mipmap that corresponds to the shape of
        # the nested tensor.
        offsets = [0]
        target_shape = image.shape[:2]
        current_shape = self.shape[:2]
        for _ in range(self.n_levels):
            if current_shape <= target_shape:
                offsets.append(
                    offsets[-1] + current_shape[0] * current_shape[1])
            current_shape = (
                current_shape[0] // self.factor,
                current_shape[1] // self.factor,
            )

        return mi.UInt32(offsets)

    def update(
        self,
        optimizer: mi.ad.Optimizer,
        parameters: mi.python.util.SceneParameters,
        iteration: int,
    ):
        if isinstance(self.nested_tensor, tensor.TensorVariable):
            super().update(optimizer, parameters, iteration)
        else:
            self.nested_tensor.update(optimizer, parameters, iteration)

    def get_value(self, optimizer: mi.ad.Optimizer):
        # Downsample optimized value for scene parameters mipmap
        image = self.nested_tensor.get_value(optimizer)

        base_shape = image.shape
        if self.shape != image.shape:
            # Find the new buffer offsets for the mipmap that corresponds to the shape
            # of the nested tensor.
            buffer_offsets = self._get_mipmap_buffer_offsets_from_tensor(image)
        else:
            if self.shape != image.shape:
                raise ValueError(
                    f'Shape of nested tensor {image.shape} does not match shape of'
                    f' highest resolution mipmap {self.shape}!'
                )
            # No need to recompute buffer offsets
            buffer_offsets = self.buffer_offsets

        n_levels = len(buffer_offsets) - 1
        buffer = dr.zeros(self.storage_type, shape=buffer_offsets[-1])
        # Scatter to level 0 (the highest resolution level)
        value = (
            image.array
            if self.storage_type == mi.Float
            else dr.unravel(self.storage_type, image.array)
        )
        index = dr.arange(
            mi.UInt32,
            start=buffer_offsets[0],
            stop=buffer_offsets[1],
        )
        dr.scatter(buffer, value, index)
        for level in range(n_levels):
            if level < n_levels - 1:
                # Downsample to next level and scatter
                image = losses._downsample_image(image)
                value = (
                    image.array
                    if self.storage_type == mi.Float
                    else dr.unravel(self.storage_type, image.array)
                )
                index = dr.arange(
                    mi.UInt32,
                    start=buffer_offsets[level + 1],
                    stop=buffer_offsets[level + 2],
                )
                dr.scatter(buffer, value, index)
                # Ensure we don't recompute image and buffer separately
                # (i.e. merge kernels)
                dr.eval(image, buffer)

        values = {}
        # miplevel and pyramid level are inverted
        values[f'{self.key}.flat_buffer'] = buffer
        values[f'{self.key}.flat_buffer_offsets'] = buffer_offsets
        values[f'{self.key}.base_mip_shape'] = mi.ScalarPoint3u(*base_shape)
        values[f'{self.key}.mip_factor'] = self.factor
        return values
