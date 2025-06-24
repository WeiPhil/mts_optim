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

"""Defines an "array" variable, see its docstring for details."""

from __future__ import annotations

from collections.abc import Mapping

import drjit as dr  # type: ignore
import mitsuba as mi  # type: ignore

from mts_optim.variables import variable


class ArrayVariable(variable.Variable):
    """An "array" optimization variable.

    This class facilitates optimization of Mitsuba/Dr.Jit array type values (e.g.,
    mi.Float, mi.Vector3f, mi.Matrix3f). For 2D/3D tensors, see the TensorVariable
    in tensor.py.
    """

    def initialize(
        self,
        optimizer: mi.ad.Optimizer,
        parameters: mi.python.util.SceneParameters,
    ):
        super().initialize(optimizer, parameters)

        value = self.initial_value
        if self.is_scene_parameter:
            if value is None:
                value = parameters[self.key]
            elif not isinstance(value, dr.ArrayBase):
                value = type(parameters[self.key])(value)
        optimizer[self.optimizer_key] = value
        optimizer.set_learning_rate(self.get_learning_rates())

    def update(
        self,
        optimizer: mi.ad.Optimizer,
        parameters: mi.python.util.SceneParameters,
        iteration: int,
    ):
        super().update(optimizer, parameters, iteration)

        if self.clamp_range is not None:
            optimizer[self.optimizer_key] = dr.clamp(
                optimizer[self.optimizer_key],
                self.clamp_range[0],
                self.clamp_range[1],
            )

    def get_value(
        self, optimizer: mi.ad.Optimizer
    ) -> Mapping[str, mi.TensorXf] | mi.TensorXf:
        if not self._registered_nested_variables:
            return optimizer[self.optimizer_key]
        values = {}
        for nested in self._registered_nested_variables:
            values |= nested.get_value(optimizer)
        values[self.optimizer_key] = optimizer[self.optimizer_key]
        return values

    def regularization_loss(self, optimizer: mi.ad.Optimizer) -> mi.Float:
        loss_sum = super().regularization_loss(optimizer)
        if self.regularization_fn is not None:
            loss_sum += self.regularization_weight * self.regularization_fn(
                self.get_value(optimizer)
            )
        return loss_sum

    def process_gradients(self, optimizer: mi.ad.Optimizer):
        super().process_gradients(optimizer)

        if self.optimizer_key not in optimizer:
            raise ValueError(
                f'ArrayVariable: {self.optimizer_key} not found in optimizer. This'
                ' likely indicates that `initialize` was not called.'
            )

        value = optimizer[self.optimizer_key]
        gradient = dr.grad(value)
        gradient = dr.select(dr.isfinite(gradient), gradient, 0.0)
        if self.gradient_clamp_range is not None:
            clamp_range = self.gradient_clamp_range
            if self.gradient_clamp_relative:
                clamp_range = clamp_range * dr.detach(dr.abs(value))
            gradient = dr.clamp(gradient, -clamp_range, clamp_range)
        dr.set_grad(value, gradient)
