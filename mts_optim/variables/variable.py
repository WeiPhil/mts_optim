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

"""Defines a base class for all optimization variables.

All optimized variables are represented using the Variable class or a class
inheriting from it. The Variable class is responsible to handle update rules,
clamping, initialization and serialization of different optimized quantities.
"""

from __future__ import annotations

import abc
from collections.abc import Callable, Mapping, Sequence

import drjit as dr  # type: ignore
import mitsuba as mi  # type: ignore
import numpy as np


class Variable(metaclass=abc.ABCMeta):
    """Represents a variable in the optimization.

    A variable is associated with one (or more) Mitsuba scene parameter and
    handles the logic to update and initialize that parameter during optimization.

    The idea is that the Variable class abstracts complex logic around
    variable initialization and updates, as well as regularization and e.g.,
    clamping.

    A variable can either represent a Mitsuba TensorXf or a low-dimensional
    value, e.g., Float, Vector3f, Color3f, etc.

    Attributes:
      key: The string key representing this variable. This corresponds to this
        variable's match in Mitsuba's `SceneParameters` object returned by
        `mi.traverse(scene)`.
      optimizer_key: The key to use in the optimizer. This is usually the same as
        the `key` if the variable is a scene parameter.
      clamp_range: Tuple of (min_value, max_value) to clamp the given parameter
        to, e.g. (0, 1).
      initial_value: The initial value of this parameter. If the Variable is a
        scene parameter (is_scene_parameter=True), the initial value will be cast
        to the appropriate type or retrieved from the scene if it is None. If the
        value is not a scene parameter, the initial value itself already needs to
        be a suitable differentiable type (e.g., mi.Vector3f).
      learning_rate: The learning rate for this specific parameter.
      learning_rate_scale: A multiplicative learning rate scale. By providing this
        as a separate argument, it is easier to separately configure the parameter
        learning rate and a global scaling factor.
      is_scene_parameter: has corresponding match in Mitsuba's `SceneParameters`
        (set to False if variable is external to the scene)
      regularization_fn: An optional regularization function.
      regularization_weight: The weight of the regularization, if a regularization
        function has been provided.
      gradient_clamp_range: The range to which to clamp the gradient to. If set to
        `None`, no clamping takes place.
      gradient_clamp_relative: Whether the clamping is done relative to the
        original value. E.g., when relative clamping is enabled, a range of "0.1"
        means the gradient will be clamped to a maximum magnitude of 0.1 * abs(X),
        where X is the current parameter value.
    """

    def __init__(
        self,
        key: str,
        *,
        optimizer_key: str | None = None,
        initial_value: (
            float | Sequence[float] | np.ndarray | dr.ArrayBase | None
        ) = None,
        clamp_range: tuple[float, float] | None = None,
        learning_rate: float | None = None,
        learning_rate_scale: float | None = None,
        is_scene_parameter: bool = True,
        regularization_fn: Callable[..., mi.Float] | None = None,
        regularization_weight: float = 1.0,
        gradient_clamp_range: float | None = None,
        gradient_clamp_relative: bool = False,
        **kwargs,
    ):
        del kwargs
        self.key = key
        self.optimizer_key = optimizer_key if optimizer_key is not None else key
        self.clamp_range = clamp_range
        self.initial_value = initial_value
        self.learning_rate = learning_rate
        if self.learning_rate is not None and learning_rate_scale is not None:
            self.learning_rate *= learning_rate_scale

        self.is_scene_parameter = is_scene_parameter
        self.regularization_fn = regularization_fn
        self.regularization_weight = regularization_weight
        self.gradient_clamp_range = gradient_clamp_range
        self.gradient_clamp_relative = gradient_clamp_relative
        self._registered_nested_variables = []

    def __repr__(self):
        to_string = (
            f'{self.__class__.__name__}(\n\t'
            f'key={self.key},\n\t'
            f' optimizer_key={self.optimizer_key},\n\t'
            f' initial_value={self.initial_value},\n\t'
            f' learning_rate={self.learning_rate},\n\t'
            f' is_scene_parameter={self.is_scene_parameter},\n\t'
            ' nested_variables=['
        )
        if not self._registered_nested_variables:
            to_string += '])'
        else:
            to_string += (
                ',\n\t'.join(
                    [str(variable)
                     for variable in self._registered_nested_variables]
                )
                + '])'
            )
        return to_string

    def get_learning_rates(self) -> dict[str, float]:
        """Returns the learning rates for this variable.

        The returned value is a dictionary, which can specify a learning rate
        for each key that this variable uses in the optimizer. E.g.,
        if the `get_value` call returns `optimizer['a'] + optimizer['b']`,
        this function might return {'a': 1.0, 'b': 2.0} to use a larger learning
        rate for the parameter 'b'. If several variables overwrite the learning
        rate for the same parameter key, the behavior is unspecified.
        """
        if self.learning_rate is not None:
            return {self.optimizer_key: self.learning_rate}
        else:
            return {}

    def initialize(
        self,
        optimizer: mi.ad.Optimizer,
        parameters: mi.python.util.SceneParameters,
    ):
        """Initialization routine for the current variable.

        Args:
          optimizer: The used optimizer. This function is expected to add its
            optimization parameters to the optimizer.
          parameters: The SceneParameters object. This is intended to only be read
            from in this call (e.g., to retrieve the initial state of a scene)
        """
        for nested in self._registered_nested_variables:
            nested.initialize(optimizer, parameters)

    def update(
        self,
        optimizer: mi.ad.Optimizer,
        parameters: mi.python.util.SceneParameters,
        iteration: int,
    ):
        """Update function to be called after optimizer.step() is invoked."""

        if self.clamp_range is not None:
            optimizer[self.optimizer_key] = dr.clamp(
                optimizer[self.optimizer_key],
                self.clamp_range[0],
                self.clamp_range[1],
            )

        for nested in self._registered_nested_variables:
            nested.update(optimizer, parameters, iteration)

    def scene_update(
        self,
        optimizer: mi.ad.Optimizer,
        parameters: mi.python.util.SceneParameters | dict[str, dr.ArrayBase],
    ):
        """Update function that is invoked to re-compute scene parameters before rendering.

        Args:
         optimizer: The Optimizer instance, which should be used to retrieve the
           current parameter values.
         parameters: The SceneParameters object to which updates have to be written
           to.
        """
        if self.is_scene_parameter:

            values = self.get_value(optimizer)

            if not isinstance(values, dict):
                values = {self.key: values}

            for key, value in values.items():
                # Special case if the optimizer holds a tensor, but we actually
                # need a Dr.Jit fixed-size array (e.g., Color3f or Matrix4f).
                if (
                    key in parameters
                    and dr.is_tensor_v(value)
                    and not dr.is_tensor_v(parameters[key])
                ):
                    parameters[key] = dr.unravel(
                        type(parameters[key]), value.array)
                # Special case if the optimizer holds a Dr.Jit fixed-size array, but we actually
                # need a Float array
                elif type(parameters[key]) != type(value):
                    parameters[key] = dr.ravel(value)
                else:
                    parameters[key] = value

    @abc.abstractmethod
    def get_value(
        self, optimizer: mi.ad.Optimizer
    ) -> Mapping[str, mi.TensorXf] | mi.TensorXf:
        """Returns the value of the optimized parameter.

        Args:
          optimizer: The Optimizer instance.

        Returns:
          The optimized parameter value.
        """

    def regularization_loss(self, optimizer: mi.ad.Optimizer) -> mi.Float:
        """Evaluates a regularizer on the given variable.

        Args:
          optimizer: The used optimizer.

        Returns:
          The evaluated regularization loss as a (differentiable) scalar.
        """

        loss_sum = mi.Float(0.0)
        for nested in self._registered_nested_variables:
            loss_sum += nested.regularization_loss(optimizer)

        if self.regularization_fn is not None:
            loss_sum += self.regularization_weight * self.regularization_fn(
                self.get_value(optimizer)
            )
        return loss_sum

    def process_gradients(self, optimizer: mi.ad.Optimizer):
        """Optional gradient post-processing step.

        This method will be called before the optimizer's update step. The
        default implementation performs simple gradient clamping. A child class
        might override this method to, e.g., implement NaN value masking.

        Args:
          optimizer: The used optimizer.
        """

        for nested in self._registered_nested_variables:
            nested.process_gradients(optimizer)

    def register_nested_variables(self, *args):
        """Registers nested variables.

        To support nesting of variables, we expect the user to explicitly
        register any nested variable objects. This has to be done in
        the __init__ call, such that subsequent calls (e.g., initialize) can
        automatically handle nested variables.

        Args:
          *args: Any nested variables may be provided directly as arguments or in a
            sequence (e.g., as a list).
        """
        for arg in args:
            if isinstance(arg, Variable):
                self._registered_nested_variables.append(arg)
            elif isinstance(arg, Sequence):
                self.register_nested_variables(*arg)
