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

import drjit as dr  # type: ignore
import mitsuba as mi  # type: ignore
import numpy as np

from mts_optim.utils import image_util
from mts_optim.utils import losses
from mts_optim.utils import pyramid
from mts_optim.variables import variable


class FlatMipAwareImagePyramidVariable(variable.Variable):
    """Represents a variable as a multiresolution pyramid and a possible associated mipmapped Bitmap.

    Attributes:
      n_levels: Levels of the multiresolution pyramid. If the number of levels is
        equal to 1, the pyramid effectively reverts to a regular 2D image.
      factor: The downsampling factor between each level of the pyramid. Has to be
        a power of 2.
      pyramid: The multiresolution pyramid itself.
      mipmapped: If True, the variable needs to be linked to a mipmapped tensor in
        the scene parameters.
      current_lod: The current LoD level. (0 = lowest resolution, -1 = highest
        resolution).
    """

    def __init__(
        self,
        key: str,
        initial_value: float | np.ndarray,
        n_levels: int | None,
        factor: int | None,
        shape: tuple[int, int, int] | None,
        mipmapped: bool = False,
        normal_clamping: bool = False,
        ensure_frequency_decomposition: bool = True,
        current_lod: int = -1,
        progressive_lr: bool = False,
        **kwargs,
    ):
        super().__init__(key=key, initial_value=initial_value, **kwargs)
        self.n_levels = None if mipmapped else n_levels
        self.factor = None if mipmapped else factor
        self.shape = None if mipmapped else shape
        self.pyramid = None
        self.mipmapped = mipmapped
        self.normal_clamping = normal_clamping
        self.ensure_frequency_decomposition = ensure_frequency_decomposition
        self.current_lod = current_lod
        self.progressive_lr = progressive_lr
        self.buffer_offsets = None
        self.storage_type = None

    def _get_level_key(self, level: int) -> str:
        return f'{self.optimizer_key}.pyramid.{level}'

    def set_current_lod(self, lod: int) -> None:
        self.current_lod = lod

    def _enable_gradient(self, optimizer: mi.ad.Optimizer):
        assert self.pyramid is not None
        for i in range(self.n_levels):
            optimizer_key = self._get_level_key(i)
            dr.enable_grad(optimizer[optimizer_key])
            self.pyramid.pyramid[i] = optimizer[optimizer_key]

    def _extract_pyramid_parameters_from_scene_parameters(
        self, scene_parameters: mi.python.util.SceneParameters
    ):
        if not self.is_scene_parameter and self.key == self.optimizer_key:
            assert self.shape is not None
            assert self.factor is not None
            assert self.n_levels is not None
            if hasattr(self.initial_value, 'shape'):
                assert self.initial_value.shape == self.shape
            return

        if f'{self.key}.base_mip_shape' not in scene_parameters.keys():
            raise ValueError(
                f'{self.key} does not represent a flattened mipmap in the scene'
                ' parameters!'
            )
        self.shape = tuple(scene_parameters[f'{self.key}' + '.base_mip_shape'])
        self.factor = scene_parameters[f'{self.key}' + '.mip_factor']
        self.buffer_offsets = scene_parameters[
            f'{self.key}' + '.flat_buffer_offsets'
        ]
        self.n_levels = len(self.buffer_offsets) - 1

        self.storage_type = None
        if self.shape[2] == 1:
            self.storage_type = mi.Float
        else:
            assert self.shape[2] == 3
            self.storage_type = mi.Color3f

        # Downsample initial value to fit the shape if it is a shape
        if hasattr(self.initial_value, 'shape'):
            print(
                f'Resizing initial value for {self.key}/{self.optimizer_key} to'
                f' shape: {self.shape}'
            )
            if self.shape[1] != self.initial_value.shape[1]:
                self.initial_value = image_util.resize_to_width(
                    self.initial_value, self.shape[1]
                )
            else:
                print('Skipping resizing as target shape is already reached.')
            assert self.initial_value.shape == self.shape
        else:
            print(
                f'Flat mip aware pyramid {self.key}/{self.optimizer_key} has shape'
                f' {self.shape}'
            )

    def initialize(
        self,
        optimizer: mi.ad.Optimizer,
        parameters: mi.python.util.SceneParameters,
    ):
        self._extract_pyramid_parameters_from_scene_parameters(parameters)
        self.pyramid = pyramid.ImagePyramid(
            self.initial_value,
            n_levels=self.n_levels,
            factor=self.factor,
            shape=(
                None
                if np.asarray(self.initial_value).size not in (1, 3)
                else self.shape
            ),
            bilinear_interpolation=True,
        )
        base_learning_rate = self.get_learning_rates()[self.optimizer_key]
        for i in range(self.n_levels):
            if self.progressive_lr:
                new_lr = base_learning_rate * (self.factor**2) ** i
            else:
                new_lr = base_learning_rate
            k = self._get_level_key(i)
            optimizer[k] = self.pyramid.pyramid[i]
            lr = {k: new_lr}
            optimizer.set_learning_rate(lr)

        self._enable_gradient(optimizer)

    def _set_pyramid_levels(self, optimizer: mi.ad.Optimizer):
        """Sets the internal state of the texture pyramid from the optimizer."""
        assert self.pyramid is not None
        for i in range(self.n_levels):
            self.pyramid.pyramid[i] = optimizer[self._get_level_key(i)]

    def clamped_frequency_decomposition(self):
        assert self.pyramid is not None
        with dr.suspend_grad():
            # Get the pyramid at the highest resolution level.
            result = self.pyramid.get_image(level=-1)

            # Apply any clamping to the highest resolution level.
            if self.normal_clamping:
                normals = dr.unravel(mi.Normal3f, result.array)
                normalized = (dr.normalize((normals * 2.0) - 1) + 1) * 0.5
                result = mi.TensorXf(dr.ravel(normalized), shape=result.shape)
            elif self.clamp_range is not None:
                result = dr.clip(
                    result, self.clamp_range[0], self.clamp_range[1])

            # Finally rebuild the pyramid from the clamped result and update the
            # pyramid.
            for level in range(self.n_levels):
                if level < self.n_levels - 1:
                    # Downsample to next level and scatter
                    next_result = losses._downsample_image(result)
                    laplacian = result - image_util.bilinear_upsample(
                        mi.Texture2f(next_result), target_shape=result.shape
                    )
                    dr.eval(laplacian, next_result)
                    self.pyramid.pyramid[(
                        self.n_levels - 1) - level] = laplacian
                    result = next_result
                    dr.schedule(
                        self.pyramid.pyramid[(self.n_levels - 1) - level])
            # Lowest level laplacian = gaussian at level 0
            self.pyramid.pyramid[0] = result
            dr.schedule(self.pyramid.pyramid[0])

    def update(
        self,
        optimizer: mi.ad.Optimizer,
        parameters: mi.python.util.SceneParameters,
        iteration: int,
    ):
        if self.pyramid is None:
            raise ValueError(
                f'Pyramid variable {self.key} has not been initialized yet!'
            )
        self._set_pyramid_levels(optimizer)
        assert self.pyramid is not None

        update_optimizer = (
            self.clamp_range is not None or self.ensure_frequency_decomposition
        )

        if self.ensure_frequency_decomposition:
            self.clamped_frequency_decomposition()
        else:
            current_lod = self.current_lod if not self.mipmapped else -1

            if self.normal_clamping:
                self.pyramid.normalize_channels(level=self.current_lod)
            elif self.clamp_range is not None:
                self.pyramid.clamp(
                    self.clamp_range[0],
                    self.clamp_range[1],
                    level=current_lod,
                    mipmap_clamping=True,
                )

        # If the pyramid was updated, we need to re-set the optimizer parameters.
        if update_optimizer:
            for i in range(self.n_levels):
                optimizer[self._get_level_key(i)] = self.pyramid.pyramid[i]

        self._enable_gradient(optimizer)

    def _infer_instance_params_from_optimizer(
        self,
        optimizer: mi.ad.Optimizer,
    ) -> None:
        """Infers the pyramid parameters from the optimizer."""
        if not self.mipmapped:
            if self.n_levels is None or self.factor is None or self.shape is None:
                raise ValueError(
                    f'Non-mipmapped pyramid with key {self.key} should have levels'
                    ' factor and shape set already!'
                )
            else:
                return

        if (
            self.n_levels is not None
            and self.factor is not None
            and self.shape is not None
        ):
            raise ValueError(
                f'Pyramid parameters for key {self.key} are already set!'
            )

        self.n_levels = 0
        while f'{self.optimizer_key}.pyramid.{self.n_levels}' in optimizer:
            self.n_levels += 1
        level_0_key = self._get_level_key(self.n_levels - 1)
        level_1_key = self._get_level_key(self.n_levels - 2)
        self.shape = optimizer[level_0_key].shape
        width_factor = self.shape[0] // optimizer[level_1_key].shape[0]
        height_factor = self.shape[1] // optimizer[level_1_key].shape[1]
        if width_factor != height_factor:
            raise ValueError(
                f'Pyramid variable {self.optimizer_key} has different width and'
                ' height factors!'
            )
        self.factor = width_factor
        # Set storage type
        if self.shape[2] == 1:
            self.storage_type = mi.Float
        else:
            assert self.shape[2] == 3
            self.storage_type = mi.Color3f

        # Build buffer offsets
        buffer_offsets = [0]
        for level in reversed(range(self.n_levels)):
            level_key = self._get_level_key(level)
            current_shape = optimizer[level_key].shape
            next_buffer_size = (
                buffer_offsets[-1] + current_shape[0] * current_shape[1]
            )
            buffer_offsets.append(next_buffer_size)
        self.buffer_offsets = mi.UInt32(buffer_offsets)

    def get_value(self, optimizer: mi.ad.Optimizer):
        if self.pyramid is None:
            self._infer_instance_params_from_optimizer(optimizer)
            self.pyramid = pyramid.ImagePyramid(
                0.0,
                n_levels=self.n_levels,
                factor=self.factor,
                shape=self.shape,
                bilinear_interpolation=True,
            )
        self._set_pyramid_levels(optimizer)
        assert self.pyramid is not None

        if not self.mipmapped:
            return self.pyramid.get_image(level=self.current_lod)
        else:
            values = {}
            # miplevel and pyramid level are inverted
            values[f'{self.key}.flat_buffer'] = self.pyramid.get_flat_image(
                self.buffer_offsets
            )
            values[f'{self.key}.flat_buffer_offsets'] = mi.UInt32(
                self.buffer_offsets)
            values[f'{self.key}.base_mip_shape'] = mi.ScalarPoint3u(
                *self.shape)
            values[f'{self.key}.mip_factor'] = self.factor
            return values

    def regularization_loss(self, optimizer: mi.ad.Optimizer) -> mi.Float:
        self._set_pyramid_levels(optimizer)
        assert self.pyramid is not None
        if self.regularization_fn is None:
            return mi.Float(0.0)
        else:
            return self.regularization_weight * self.regularization_fn(self.pyramid)

    def process_gradients(self, optimizer: mi.ad.Optimizer):
        super().process_gradients(optimizer)

        for i in range(self.n_levels):
            value = optimizer[self._get_level_key(i)]
            gradient = dr.grad(value)
            gradient = dr.select(dr.isfinite(gradient), gradient, 0.0)

            if self.gradient_clamp_range is not None:
                clamp_range = self.gradient_clamp_range
                if self.gradient_clamp_relative:
                    clamp_range = clamp_range * dr.detach(dr.abs(value))
                gradient = dr.clamp(gradient, -clamp_range, clamp_range)
            dr.set_grad(value, gradient)
