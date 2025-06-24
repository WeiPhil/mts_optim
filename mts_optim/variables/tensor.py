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

"""Defines a tensor variable that represents images or volume grids."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import drjit as dr  # type: ignore
import mitsuba as mi  # type: ignore
import numpy as np

from mts_optim.utils import image_util
from mts_optim.variables import array

_INITIAL_VALUE_TYPE = float | Sequence[float] | np.ndarray | dr.ArrayBase


class TensorVariable(array.ArrayVariable):
    """Represents a tensorial variable in the optimization.

    This Variable represents a Tensor-valued optimization variable. It supports
    broadcasting of initial values to a target shape and upsampling during
    optimization.

    Attributes:
      key: The string key representing this variable. This corresponds to this
        variable's match in Mitsuba's `SceneParameters` object returned by
        `mi.traverse(scene)`.
      initial_value: The initial value of this parameter. If a shape is provided,
        the initial value will be broadcast over the last dimension. E.g.,
        providing an initial value of (0.5,1.0,0.5) and a shape of (64,64,3) will
        initialize the variable to be an image with RGB values (0.5,1.0,0.5) in
        each pixel. The initial value can either be a NumPy array or Mitsuba
        TensorXf, a non-tensor Dr.Jit type (e.g., mi.ScalarVecto3f) or simply a
        sequence of floats (e.g., a 3-tuple to specify a color).
      shape: The shape of this parameter (e.g., (512,512,3) for an RGB texture).
      upsampling_schedule: Optionally, a schedule to upsample the variable during
        the optimization.
    """

    def _check_shape(
        self, shape: tuple[int, ...], initial_value: _INITIAL_VALUE_TYPE
    ):
        """Validates the compatibility of shape and initial value.

        This ensures that the initial value either already has the same shape
        as the provided target shape, or is broadcastable to the target shape.

        Args:
          shape: The specified Variable shape.
          initial_value: The initial value.

        Raises:
          ValueError if the shape and initial value are incompatible.
        """
        channels = shape[-1]
        if isinstance(initial_value, np.ndarray | mi.TensorXf):
            n_elements = np.prod(initial_value.shape)
            if n_elements != channels and initial_value.shape != shape:
                raise ValueError(
                    f'Expected shape {shape} or ({channels},) but got'
                    f' {initial_value.shape}'
                )
        elif isinstance(initial_value, Sequence):
            if len(initial_value) != channels:
                raise ValueError(
                    f'Expected shape {shape} or ({channels},) but got'
                    f' {len(initial_value)}'
                )
        elif isinstance(initial_value, dr.ArrayBase) and not dr.is_tensor_v(
            initial_value
        ):
            # Check for the size of non-tensor Dr.Jit values.
            if initial_value.ndim > 1:
                raise ValueError(
                    'Higher-order Dr.Jit non-Tensor types are currently not supported'
                    ' as initial values. Consider providing the initial value as NumPy'
                    ' array or Dr.Jit scalar type instead (e.g., mi.ScalarVector3f).'
                )
            if initial_value.shape[0] != channels:
                raise ValueError(
                    f'Expected shape ({channels},) but got {initial_value.shape[0]}'
                )

    def __init__(
        self,
        key: str,
        *,
        initial_value: _INITIAL_VALUE_TYPE | None = None,
        shape: tuple[int, ...] | None = None,
        upsampling_schedule: Callable[[int], int] | None = None,
        **kwargs,
    ):
        super().__init__(key=key, **kwargs)
        del kwargs

        if shape is not None:
            if not self.is_scene_parameter and initial_value is None:
                raise ValueError(
                    'Initial value must be provided when a shape is specified and'
                    ' is_scene_parameter=False.'
                )
            self._check_shape(shape, initial_value)
            self.shape = shape
        elif isinstance(initial_value, np.ndarray | mi.TensorXf):
            self.shape = initial_value.shape
        else:
            raise ValueError(
                'A shape must explicitly be provided for initial values that are not'
                ' a NumPy array or Mitsuba Tensor.'
            )

        if upsampling_schedule is not None:
            # Check number of unique elements.
            if len(set(self.shape[:-1])) != 1:
                raise ValueError(
                    f'Invalid shape {self.shape}: Upsampling is currently only'
                    ' supported for tensors with all equal side lengths.'
                )

        self.initial_value = initial_value
        self.upsampling_schedule = upsampling_schedule

    def _update_initial_values_from_upsampling_schedule(self):
        """Updates the initial values based on the upsampling schedule if needed."""

        if self.upsampling_schedule is None:
            return

        initial_size = self.upsampling_schedule(0)

        # Special case for 2D images: If needed, resample to initial resolution.
        if (
            isinstance(self.initial_value, np.ndarray | mi.TensorXf)
            and self.initial_value.ndim == 3
        ):
            self.initial_value = image_util.resize_to_width(
                self.initial_value, initial_size
            )

        self.shape = (len(self.shape) - 1) * \
            (initial_size,) + (self.shape[-1],)
        if len(self.shape) <= 1 or len(self.shape) > 4:
            raise ValueError(
                f'Length of shape tuple needs to be in {2, 3, 4}, but got'
                f' {len(self.shape)}.'
            )

    def initialize(
        self,
        optimizer: mi.ad.Optimizer,
        parameters: mi.python.util.SceneParameters,
    ):
        """Initializes the Variable and adds it to the optimizer.

        Args:
          optimizer: The used optimizer. This function is expected to add its
            optimization parameters to the optimizer.
          parameters: The SceneParameters object. This is intended to only be read
            from in this call (e.g., to retrieve the initial state of a scene)
        """

        self._update_initial_values_from_upsampling_schedule()

        if self.is_scene_parameter:
            parameter_type = type(parameters[self.key])
            if not dr.is_tensor_v(parameter_type):
                raise ValueError(
                    f'The associated parameter of key {self.key} is of type'
                    f' {parameter_type} and not a tensor!'
                )

        value = self.initial_value
        is_array_type = isinstance(value, np.ndarray | mi.TensorXf)
        needs_broadcasting = not (is_array_type and value.shape == self.shape)
        if needs_broadcasting:
            value = np.ones(self.shape) * np.array(value).ravel()
            # Since we pre-validate input sizes, this should now always hold.
            assert value.shape == self.shape

        if isinstance(value, np.ndarray):
            value = mi.TensorXf(value)

        optimizer[self.optimizer_key] = value
        optimizer.set_learning_rate(self.get_learning_rates())

    def update(
        self,
        optimizer: mi.ad.Optimizer,
        parameters: mi.python.util.SceneParameters,
        iteration: int,
    ):
        """Updates the Variable after an optimization step."""

        # If needed, upsample the underlying texture.
        if self.upsampling_schedule is not None:
            size = self.upsampling_schedule(iteration)
            if size > self.shape[0]:
                dimension = len(self.shape) - 1
                texture_types = ('None', mi.Texture1f,
                                 mi.Texture2f, mi.Texture3f)
                texture = texture_types[dimension](
                    optimizer[self.optimizer_key])
                target_shape = (size,) * (dimension) + (self.shape[-1],)
                optimizer[self.optimizer_key] = dr.upsample(
                    texture, shape=target_shape
                ).tensor()
                self.shape = target_shape

        super().update(optimizer, parameters, iteration)
