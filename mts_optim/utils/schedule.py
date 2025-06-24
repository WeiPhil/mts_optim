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

"""Implements different "schedule" functions, e.g. for learning rate decay."""

from collections.abc import Callable

import numpy as np


def exponential(
    target_value: int | float,
    factor: int | float,
    n_steps: int,
    n_iterations: int,
) -> Callable[[int], int | float]:
  """Returns an exponential schedule.

  This schedule multiplies an initial value by a scale factor a number of
  n_steps-1 times. Morever, the number of iterations between successive changes
  increase exponentially (by a factor of 2). The schedule is exponential both in
  the change of the initial value and the increasing gaps between successive
  changes.

  For example, for a target value of 64 and n_iterations=8, a schedule with
  factor=4 and n_steps=3, would produce the value of "4" for 2 iterations, "16"
  for 2 iterations and 64 for the remaining 4 iterations:

  Iteration: 0  1  2  3  4  5  6  7
      Value: 4  4 16 16 64 64 64 64

  This function returns a "schedule", which is a function that can be invoked
  with the current iteration count. The schedule function assumes that
  the user invokes it with a valid iteration 0 <= count < n_iterations.

  Args:
    target_value: The final target value to attain. This can either be an
      integer or a float.
    factor: The factor by which to scale the value between each step.
    n_steps: The number of distinct steps to consider. Note that the last step
      will always be the target value (i.e., if n_steps=1 this schedule simply
      returns the target value for all iterations).
    n_iterations: The number of iterations for this schedule.

  Returns:
    A function that returns the current value for a given iteration i.

  Raises:
    ValueError: If n_iterations <= 0
  """

  if n_iterations <= 0:
    raise ValueError('n_iterations must be strictly positive.')

  # Internally simply precompute an array of values attained in each iteration.
  values = np.full(n_iterations, target_value, dtype=type(target_value))
  previous = n_iterations // 2
  current = previous // 2
  value = target_value
  for i in range(n_steps - 1):
    if i == n_steps - 2:  # Special case for the starting iterations.
      current = 0
    if isinstance(target_value, int):
      value = value // factor
    else:
      value = value / factor
    values[current:previous] = value
    previous = current
    current = current // 2
  return (
      lambda i: type(target_value)(values[i])
      if i < n_iterations
      else type(target_value)(values[-1])
  )


def step(
    steps: list[tuple[float, float | int]], n_iterations: int
) -> Callable[[int], int | float]:
  """Returns a step schedule.

  This schedule is a piecewise constant schedule. The schedule is determined
  by a list of 'steps'. Each 'step' in that list is a pair, where the first
  element is the floating point value corresponding to the relative progress at
  which the step should take place (e.g., 0.3 = 30%) and the second element is
  the actual value.

  This allows to specify a step schedule relative to the total number of
  iterations. E.g., steps = [(0.0, 10.0), (0.5, 2.0), (0.8, 0.2)] will return
  a value of 10.0 until 50% of the iterations were completed, 2.0 between 50 and
  80% and 0.2 for the remainder.

  Args:
    steps: A list of pairs (progress, value), where the progress values need to
      be strictly increasing, and the first element has to be 0.0 (i.e., the
      start of the schedule).
    n_iterations: The total number of iterations.

  Returns:
    A function that returns the current value for a given iteration i.
  """

  if n_iterations <= 0:
    raise ValueError('n_iterations must be strictly positive.')

  # Validate the input to be a strictly increasing sequence of steps.
  if steps[0][0] != 0:
    raise ValueError('First step must be at 0.0!')
  previous_progress = steps[0][0]
  for progress, _ in steps[1:]:
    if not progress > previous_progress:
      raise ValueError('Steps must be strictly increasing')
    if progress >= 1.0:
      raise ValueError('Steps must be between 0 and 1 (exclusive)!')
    previous_progress = progress

  def step_schedule(i):
    previous = None
    for progress, value in steps:
      if progress * n_iterations > i:
        break
      previous = value
    assert previous is not None
    return previous

  return step_schedule


def power(
    n_iterations: int,
    n_steps: int,
    exponent: float = 0.5,
) -> Callable[[int], int]:
  """Returns a monotonic schedule from a power function.

  This schedule scales the power function to gain the n_step value at
  n_iterations. It evaluates the step number by rounding down to an integer.
  The exponent must be non-negative. The following behaviour is guaranteed for
  the following ranges of exponent:
  0: Only outputs the last step. Constant schedule.
  (0, 1): The difference between successive steps is increasing/non-decreasing,
  last step is the longest. Greater values make duration of consecutive steps
  more even.
  1: All steps have approximately the same duration. Linear schedule.
  (1, +inf): The difference between successive steps is decreasing/non-
  increasing. First step is the longest. Smaller values make duration of
  consecutive steps more even.

  Schedule always returns the last step for iteration query beyond n_iterations.

  Args:
    n_iterations: The total number of iterations.
    n_steps: The number of distinct steps.
    exponent: The exponent of the power function, must be non-negative.

  Returns:
    A function that returns the current step index for a given iteration.
  """
  if n_iterations < 1:
    raise ValueError('n_iterations must be at least 1.')
  if n_steps < 1:
    raise ValueError('n_steps must be at least 1.')
  if exponent < 0:
    raise ValueError('exponent must be non-negative.')
  elif exponent == 0:
    step_fn = lambda i: n_steps - 1
  else:
    step_fn = lambda i: int(
        np.floor(np.power(i / n_iterations, exponent) * n_steps)
    )

  def _schedule(i):
    if i < 0:
      raise ValueError('Iteration i must be at least 0.')
    if i < n_iterations:
      return step_fn(i)
    else:
      return n_steps - 1

  return _schedule
