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

from mts_optim.utils import pyramid
from mts_optim.variables import variable


class ImagePyramidVariable(variable.Variable):
    """A multiresolution pyramid variable possibly associated with a mipmap.

    Attributes:
      n_levels: Number of levels of the multiresolution pyramid. If equals 1, the
        pyramid effectively reverts to a regular 2D image.
      factor: The downsampling factor between levels of the pyramid. Has to be a
        power of 2.
      image_pyramid: The multiresolution image pyramid object.
      is_mipmapped: If True, the variable is linked to a mipmapped tensor in the
        scene parameters.
      current_level: The current level of detail. (0 for lowest resolution, -1 for
        highest resolution).
    """

    def __init__(
        self,
        key: str,
        *,
        n_levels: int | None = None,
        factor: int = 2,
        min_resolution: int | None = None,
        shape: tuple[int, int, int] | None = None,
        mipmapped: bool = False,
        normalize_channels: bool = False,
        current_level: int = -1,
        **kwargs,
    ):
        super().__init__(key=key, **kwargs)
        self.n_levels = n_levels
        self.factor = factor
        self._min_resolution = min_resolution
        self._initialization_shape = shape
        self.is_mipmapped = mipmapped
        self.normalize_channels = normalize_channels
        self._current_level = current_level

        if self.normalize_channels and self.clamp_range is not None:
            raise ValueError(
                'Only one of `clamp_range` and `normalize_channels` can be set.'
            )

        self.image_pyramid = None
        self._level_offsets = None

    def _get_level_key(self, level: int) -> str:
        # Remove the '.data' suffix.
        texture_key = self.optimizer_key.rsplit('.', 1)[0]
        return f'{texture_key}.pyramid.{level}'

    def set_current_level(self, level: int) -> None:
        self._current_level = level

    def _enable_gradient(self, optimizer: mi.ad.Optimizer) -> None:
        if self.image_pyramid is None:
            raise ValueError('Image pyramid is not initialized.')
        for level in range(self.n_levels):
            optimizer_key = self._get_level_key(level)
            dr.enable_grad(optimizer[optimizer_key])
            self.image_pyramid.pyramid[level] = optimizer[optimizer_key]

    def initialize(
        self,
        optimizer: mi.ad.Optimizer,
        parameters: mi.python.util.SceneParameters,
    ) -> None:
        self.image_pyramid = pyramid.ImagePyramid(
            value=self.initial_value,
            n_levels=self.n_levels,
            factor=self.factor,
            min_resolution=self._min_resolution,
            shape=self._initialization_shape,
            bilinear_interpolation=True,
        )
        if self.n_levels is None:
            self.n_levels = self.image_pyramid.n_levels
        assert self.n_levels == self.image_pyramid.n_levels

        level_offsets = [0]
        for level_image in reversed(self.image_pyramid.pyramid):
            level_size = level_image.shape[0] * level_image.shape[1]
            level_offsets.append(level_offsets[-1] + level_size)
        self._level_offsets = mi.UInt32(level_offsets)

        for level in range(self.n_levels):
            level_key = self._get_level_key(level)
            optimizer[level_key] = self.image_pyramid.pyramid[level]

        optimizer.set_learning_rate(self.get_learning_rates())
        self._enable_gradient(optimizer)

        if self.is_scene_parameter and self.is_mipmapped:
            # Remove the '.data' suffix.
            texture_key = self.key.rsplit('.', 1)[0]
            if self.factor != 2:
                raise ValueError(
                    'Only a factor of 2 is supported for mipmapped textures, got'
                    f' {self.factor}.'
                )
            base_shape = self.image_pyramid.pyramid[-1].shape
            parameters[f'{texture_key}.base_mip_shape'] = base_shape
            parameters[f'{texture_key}.level_offsets'] = self._level_offsets

    def _set_pyramid_levels(self, optimizer: mi.ad.Optimizer) -> None:
        """Sets the internal state of the texture pyramid from the optimizer."""
        if self.image_pyramid is None:
            self.image_pyramid = pyramid.ImagePyramid(
                value=0.0,
                n_levels=self.n_levels,
                factor=self.factor,
                shape=self._initialization_shape,
                bilinear_interpolation=True,
            )
        for level in range(self.n_levels):
            self.image_pyramid.pyramid[level] = optimizer[self._get_level_key(
                level)]

    def update(
        self,
        optimizer: mi.ad.Optimizer,
        parameters: mi.python.util.SceneParameters,
        iteration: int,
    ) -> None:
        if self.image_pyramid is None:
            raise ValueError('Image pyramid is not initialized.')
        if self.clamp_range is not None or self.normalize_channels:
            max_clamp_level = -1 if self.is_mipmapped else self._current_level

            # Set the pyramid levels from the optimizer to clamp or normalize them.
            self._set_pyramid_levels(optimizer)

            if self.normalize_channels:
                self.image_pyramid.normalize_channels(level=max_clamp_level)
            elif self.clamp_range is not None:
                self.image_pyramid.clamp(
                    self.clamp_range[0],
                    self.clamp_range[1],
                    level=max_clamp_level,
                    mipmap_clamping=self.is_mipmapped,
                )

            # Reset the optimizer parameters after clamping.
            for level in range(self.n_levels):
                level_key = self._get_level_key(level)
                optimizer[level_key] = self.image_pyramid.pyramid[level]

        self._enable_gradient(optimizer)

    def get_value(self, optimizer: mi.ad.Optimizer) -> mi.TensorXf:
        self._set_pyramid_levels(optimizer)
        if self.image_pyramid is None:
            raise ValueError('Image pyramid is not initialized.')

        if self.is_mipmapped:
            if self._level_offsets is None:
                raise ValueError('Level offsets are not initialized.')

            # Mip level and pyramid level are inverted.
            return self.image_pyramid.get_flat_image(self._level_offsets)

        # If not mipmapped, return the bitmap data directly.
        return self.image_pyramid.get_image(level=self._current_level)

    def regularization_loss(self, optimizer: mi.ad.Optimizer) -> mi.Float:
        self._set_pyramid_levels(optimizer)
        if self.image_pyramid is None:
            raise ValueError('Image pyramid is not initialized.')
        if self.regularization_fn is None:
            return mi.Float(0.0)
        else:
            loss = self.regularization_fn(self.image_pyramid)
            return self.regularization_weight * loss

    def process_gradients(self, optimizer: mi.ad.Optimizer) -> None:
        super().process_gradients(optimizer)

        for level in range(self.n_levels):
            value = optimizer[self._get_level_key(level)]
            gradient = dr.grad(value)
            gradient = dr.select(dr.isfinite(gradient), gradient, 0.0)
            dr.set_grad(value, gradient)

    def get_learning_rates(self) -> dict[str, float]:
        if self.learning_rate is not None:
            return {
                self._get_level_key(i): self.learning_rate
                for i in range(self.n_levels)
            }
        else:
            return {}
