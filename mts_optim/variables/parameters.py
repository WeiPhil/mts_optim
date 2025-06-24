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

"""Bundles the logic to update parameters during optimization."""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any

import drjit as dr  # type: ignore
import mitsuba as mi  # type: ignore

from mts_optim.variables import variable_types


class Variables(metaclass=abc.ABCMeta):
    """Represents a collection of Mitsuba Variables that are optimized.

    This class serves as the "glue" to connect invididual Mitsuba "Variables"
    with the used optimizer / optimization framework.

    The individual Variable classes (see the subfolder "./variables") implement
    custom Mitsuba logic which is tightly coupled to the used Mitsuba scene.
    An example of that is a Variable implementing a multiscale texture pyramid,
    for which Mitsuba does not provide built-in support.

    This class is a wrapper to transparently handle a collection of such
    Variable instances. Moreover, this collection of Variable instances might
    interact wwith different optimization frameworks (e.g., Tensorflow or
    Mitsuba's built-in optimizer classes.)

    For example uses of this class, see the optimization problems specified
    in the file `problem.py`.

    Attributes:
      variables: The optimization variables.
      scene_parameters: The scene parameters associated with the variables.
      is_initialized: A bool that tracks whether or not the variables have been
        initialized. This is required, as initialization requires an optimization
        object which we do not want to keep as internal state in this class.
    """

    def __init__(
        self,
        variables: Sequence[variable_types.Variable],
        parameters: mi.SceneParameters,
    ):
        """Initializes a list of Mitsuba optimization variables.

        Args:
          variables: A collection of Mitsuba optimization variables. Each variable
            handles the logic to update and initialize that parameter during
            optimization.
          parameters: Mitsuba scene parameters.
        """
        self._variables = variables
        self.scene_parameters = parameters
        self.is_initialized = False

    def __repr__(self):
        return ' '.join([str(variable) for variable in self._variables])

    def _check_if_initialized(self):
        if not self.is_initialized:
            raise ValueError(
                'Variables must be initialized before they can be used.')

    @abc.abstractmethod
    def initialize(self, state: Any) -> None:
        """Initializes the Variable instances.

        This function must be called prior to starting an optimization loop. It
        will write initial parameter values to the `state` object that is passed
        as an argument. The Variables class itself (or its child classes) do not
        internally keep variable values, but rather interact with concrete values
        through a state object.

        Args:
          state: The optimization state. Depending on the concrete implementation,
            this might be a Mitsuba optimizer or a Tensorflow optimizer instance.
            The result of the initialization will be written into this state object.
        """

    @abc.abstractmethod
    def update(self, state: Any, iteration: int) -> None:
        """Updates the variables and scene parameters at the current iteration.

        This is a non-differentiable operation invoked after the gradient updates
        were applied. It will call the `update` method of individual scene
        variables. This method may internally resize variables or clamp them to
        a valid range.

        Args:
          state: The optimization state. Depending on the concrete implementation,
            this might be a Mitsuba optimizer or a Tensorflow optimizer instance.
            This method will update values that are held by this state object.
          iteration: The current iteration number.

        Raises:
          ValueError: If the variables were not yet initialized.
        """

    def process_gradients(self, state: Any) -> None:
        """Processes the gradients at the current iteration.

        This calls each Variable's `process_gradients` function, which can
        for example implement gradient clamping or other postprocessing steps.

        Args:
          state: The current gradient state. The implementation will highly depend
            on the used optimization framework (i.e., Mitsuba or Tensorflow), since
            they support gradient clamping in a different way.

        Raises:
          ValueError: If the variables were not yet initialized.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def evaluate_regularization_gradients(self, state: Any) -> None:
        """Evaluates the gradients of regularization terms.

        This call evaluates the regularization terms associated with different
        variables and backpropagates their gradients.

        Args:
          state: The optimization state. Depending on the concrete implementation,
            this might be a Mitsuba optimizer or a list of Tensorflow tensors.

        Raises:
          ValueError: If the variables were not yet initialized.
        """

    def __iter__(self):
        self._variable_iterator = iter(self._variables)
        return self

    def __next__(self):
        return next(self._variable_iterator)


class MitsubaVariables(Variables):
    """A collection of Variables that can be optimized using a Mitsuba optimizer.

    This class is a collection of Mitsuba-defined Variables that can be
    optimized using a Mitsuba Optimizer. This class is intended to be used
    for optimizations that only use Mitsuba/DrJit, and no other ML framework.
    """

    def initialize(self, state: mi.ad.Optimizer) -> None:
        for variable in self._variables:
            variable.initialize(state, self.scene_parameters)
            variable.scene_update(state, self.scene_parameters)
            self.scene_parameters.update()
        self.is_initialized = True

    def update(self, state: mi.ad.Optimizer, iteration: int) -> None:
        self._check_if_initialized()
        for variable in self._variables:
            variable.update(state, self.scene_parameters, iteration)
            variable.scene_update(state, self.scene_parameters)
        self.scene_parameters.update()

    def process_gradients(self, state: mi.ad.Optimizer) -> None:
        self._check_if_initialized()
        for variable in self._variables:
            variable.process_gradients(state)

    def evaluate_regularization_gradients(self, state: mi.ad.Optimizer) -> None:
        self._check_if_initialized()
        for v in self._variables:
            regularization_loss = v.regularization_loss(state)
            if dr.grad_enabled(regularization_loss):
                dr.backward(regularization_loss)
