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

"""Implements various common image-based loss functions."""

from __future__ import annotations

from collections.abc import Callable
import functools

import drjit as dr  # type: ignore
import gin
import mitsuba as mi  # type: ignore


_mean = functools.partial(dr.mean, axis=None)


@gin.register
def l1_norm(
    image: mi.TensorXf,
    reference: mi.TensorXf,
    weight: mi.TensorXf | None = None,
    reduce_fn: Callable[[mi.TensorXf], mi.Float] = _mean,
    relative: bool = False,
    relative_denominator_epsilon: float = 1e-2,
) -> mi.Float | mi.TensorXf:
  """Computes an L1 loss.

  Args:
    image: The input image.
    reference: The known reference image.
    weight: Optionally, an element-wise weight for the computed error terms.
      This has to be a tensor that broadcasts to the shape of the input image
      (default: None).
    reduce_fn: The reduction function to be used. If None, the raw per-pixel
      loss values are returned.
    relative: Whether to evaluate the L1 norm relative to the reference image.
      The relative L1 error is defined as: relative_l1 = |image - reference| /
      (|reference| + ε).
    relative_denominator_epsilon: The epsilon value used in the division when a
      relative loss is used.

  Returns:
    The average L1 loss.
  """
  loss = dr.abs(image - reference)
  if weight is not None:
    loss *= weight

  if relative:
    loss = loss / (dr.abs(reference) + relative_denominator_epsilon)

  return reduce_fn(loss) if reduce_fn is not None else loss


@gin.register
def l2_norm(
    image: mi.TensorXf,
    reference: mi.TensorXf,
    weight: mi.TensorXf | None = None,
    reduce_fn: Callable[[mi.TensorXf], mi.Float] = _mean,
) -> mi.Float | mi.TensorXf:
  """Computes a squared L2 loss.

  Args:
    image: The input image.
    reference: The known reference image.
    weight: Optionally, an element-wise weight for the computed error terms.
      This has to be a tensor that broadcasts to the shape of the input image
      (default: None).
    reduce_fn: The reduction function to be used. If None, the raw per-pixel
      loss values are returned.

  Returns:
    The average squared L2 loss.
  """

  loss = dr.square(image - reference)
  if weight is not None:
    loss *= weight

  return reduce_fn(loss) if reduce_fn is not None else loss


@gin.configurable(denylist=['image', 'reference', 'weight'])
def huber(
    image: mi.TensorXf,
    reference: mi.TensorXf,
    weight: mi.TensorXf | None = None,
    delta: float = 0.05,
    reduce_fn: Callable[[mi.TensorXf], mi.Float] = _mean,
) -> mi.Float | mi.TensorXf:
  """Computes a Huber loss.

  The Huber loss uses an L2 loss for residuals smaller than some threshold
  `delta` and an L1 loss for any residuals larger than delta. The loss is scaled
  such that it remains continuous at the transition between L2 and L1 error.

  Args:
    image: The input image.
    reference: The known reference image.
    weight: Optionally, an element-wise weight for the computed error terms.
      This has to be a tensor that broadcasts to the shape of the input image
      (default: None).
    delta: The threshold at which the implementation switches between L2 and L1
      loss.
    reduce_fn: The reduction function to be used. If None, the raw per-pixel
      loss values are returned.

  Returns:
    The average Huber loss.
  """
  residual = image - reference
  loss = dr.select(
      dr.abs(residual) <= delta,
      0.5 * dr.square(residual),
      delta * (dr.abs(residual) - 0.5 * delta),
  )
  if weight is not None:
    loss *= weight
  return reduce_fn(loss) if reduce_fn is not None else loss


@gin.configurable(denylist=['image', 'reference', 'weight'])
def smape(
    image: mi.TensorXf,
    reference: mi.TensorXf,
    weight: mi.TensorXf | None = None,
    relative_denominator_epsilon: float = 1e-2,
    reduce_fn: Callable[[mi.TensorXf], mi.Float] = _mean,
) -> mi.Float | mi.TensorXf:
  """Computes the symmetric mean absolute percentage error (SMAPE).

  SMAPE is a relative metric which can provide an advantage over the
  standard L1 norm if the signal has high dynamic range.

  Args:
    image: The input image.
    reference: The known reference image.
    weight: Optionally, an element-wise weight for the computed error terms.
      This has to be a tensor that broadcasts to the shape of the input image
      (default: None).
    relative_denominator_epsilon: Value added to the denominator.
    reduce_fn: The reduction function to be used. If None, the raw per-pixel
      loss values are returned.

  Returns:
    The SMAPE loss using reduce_fn.
  """
  numerator = dr.abs(image - reference)
  denominator = dr.abs(image) + dr.abs(reference) + relative_denominator_epsilon
  loss = numerator / denominator
  if weight is not None:
    loss *= weight
  return reduce_fn(loss)


@gin.configurable(denylist=['image', 'reference', 'weight'])
def relative_l2_norm(
    image: mi.TensorXf,
    reference: mi.TensorXf,
    weight: mi.TensorXf | None = None,
    relative_denominator_epsilon: float = 1e-2,
    reduce_fn: Callable[[mi.TensorXf], mi.Float] = _mean,
) -> mi.Float | mi.TensorXf:
  """Computes the relative squared L2 loss.

  This implements (image - reference)**2 / (image**2 + reference**2 + epsilon).

  Args:
    image: The input image.
    reference: The known reference image.
    weight: Optionally, an element-wise weight for the computed error terms.
      This has to be a tensor that broadcasts to the shape of the input image
      (default: None).
    relative_denominator_epsilon: Value added to the denominator.
    reduce_fn: The reduction function to be used. If None, the raw per-pixel
      loss values are returned.

  Returns:
    The average squared L2 loss.
  """
  numerator = dr.square(image - reference)
  denominator = (
      dr.square(image) + dr.square(reference) + relative_denominator_epsilon
  )
  loss = numerator / denominator
  if weight is not None:
    loss *= weight
  return reduce_fn(loss)


def _srgb_transfer(x: dr.ArrayBase) -> dr.ArrayBase:
  """Differentiable computation of the srgb transfer function.

  Args:
    x: The input array, in 0-1 range.

  Returns:
    The tonemapped array.
  """
  return dr.select(
      x > 0.0031308,
      dr.power(dr.maximum(x, 0.0031308), 1.0 / 2.4) * 1.055 - 0.055,
      x * 12.92,
  )


def _log_srgb_tonemap(x: dr.ArrayBase) -> dr.ArrayBase:
  """Differentiable of the log tonemapping operator.

  Suggested in "Extracting Triangular 3D Models, Materials, and Lighting From
  Images" (Munkberg et al., 2021, Eq. 7).

  Args:
    x: The input array, in 0-1 range.

  Returns:
    The array mapped to srgb.
  """
  return _srgb_transfer(dr.log(x + 1.0)) * 255


@gin.configurable(denylist=['image', 'reference', 'weight'])
def tonemapped_smape(
    image: mi.TensorXf,
    reference: mi.TensorXf,
    weight: mi.TensorXf | None = None,
    relative_denominator_epsilon: float = 1e-2,
    reduce_fn: Callable[[mi.TensorXf], mi.Float] = _mean,
) -> mi.Float | mi.TensorXf:
  """Computes the tonemapped symmetric mean absolute percentage error (SMAPE).

  This metric is compute by applying the sRGB transfer to the log-mapped input
  values (in 0-1 range).

  SMAPE is a relative metric which can provide an advantage over the
  standard L1 norm if the signal has high dynamic range.

  Args:
    image: The input image.
    reference: The known reference image.
    weight: Optionally, an element-wise weight for the computed error terms.
      This has to be a tensor that broadcasts to the shape of the input image
      (default: None).
    relative_denominator_epsilon: Value added to the denominator.
    reduce_fn: The reduction function to be used. If None, the raw per-pixel
      loss values are returned.

  Returns:
    The SMAPE loss (tonemapped), after applying reduce_fn.
  """
  image = _log_srgb_tonemap(image)
  reference = _log_srgb_tonemap(reference)

  return smape(
      image, reference, weight, relative_denominator_epsilon, reduce_fn
  )


@gin.configurable(denylist=['image', 'reference', 'weight'])
def tonemapped_relative_l2_norm(
    image: mi.TensorXf,
    reference: mi.TensorXf,
    weight: mi.TensorXf | None = None,
    relative_denominator_epsilon: float = 1e-2,
    reduce_fn: Callable[[mi.TensorXf], mi.Float] = _mean,
) -> mi.Float | mi.TensorXf:
  """Computes the tonemapped relative squared L2 loss.

  This metric is compute by applying the sRGB transfer to the log-mapped input
  values (in 0-1 range).

  Args:
    image: The input image.
    reference: The known reference image.
    weight: Optionally, an element-wise weight for the computed error terms.
      This has to be a tensor that broadcasts to the shape of the input image
      (default: None).
    relative_denominator_epsilon: Value added to the denominator.
    reduce_fn: The reduction function to be used. If None, the raw per-pixel
      loss values are returned.

  Returns:
    The average squared L2 loss (tonemapped), after applying reduce_fn.
  """
  image = _log_srgb_tonemap(image)
  reference = _log_srgb_tonemap(reference)

  return relative_l2_norm(
      image, reference, weight, relative_denominator_epsilon, reduce_fn
  )


def _downsample_image(image: mi.TensorXf) -> mi.TensorXf:
  """Differentiable 2x image downsampling by averaging.

  Args:
    image: The input image.

  Returns:
    The downsampled image.
  """

  height, width, n_channels = image.shape[0], image.shape[1], image.shape[2]
  new_shape = (height // 2, width // 2, n_channels)
  i, j = dr.meshgrid(
      dr.arange(mi.UInt32, new_shape[0]),
      dr.arange(mi.UInt32, new_shape[1]),
      indexing='ij',
  )
  i = 2 * i
  j = 2 * j
  i = dr.repeat(i, n_channels)
  j = dr.repeat(j, n_channels)
  c = dr.tile(dr.arange(mi.Int32, n_channels), new_shape[0] * new_shape[1])
  source = image.array
  i0 = i * width
  i1 = dr.minimum(i + 1, height - 1) * width
  j0 = j
  j1 = dr.minimum(j + 1, width - 1)
  average = 0.25 * (
      dr.gather(mi.Float, source, dr.fma(i0 + j0, n_channels, c))
      + dr.gather(mi.Float, source, dr.fma(i1 + j0, n_channels, c))
      + dr.gather(mi.Float, source, dr.fma(i0 + j1, n_channels, c))
      + dr.gather(mi.Float, source, dr.fma(i1 + j1, n_channels, c))
  )
  return mi.TensorXf(average, shape=new_shape)


@gin.configurable(denylist=['image', 'reference'])
def multiscale_loss(
    image: mi.TensorXf,
    reference: mi.TensorXf,
    weight: mi.TensorXf | None = None,
    loss_fn: Callable[
        [mi.TensorXf, mi.TensorXf, mi.TensorXf | None], mi.TensorXf
    ] = l1_norm,
    n_levels: int = 4,
) -> mi.Float:
  """Returns a loss evaluated at multiple resolutions.

  This loss computes a pixel-wise loss at multiple resolutions and sums up
  the result. The benefit of this is that it increase the image-space "range"
  over which we get meaningful gradients.

  Args:
    image: The input image.
    reference: The known reference image.
    weight: Optionally, an element-wise weight for the computed error terms.
      This has to be a tensor that broadcasts to the shape of the input image
      (default: None).
    loss_fn: The loss function to be used at each level.
    n_levels: The number of resolution levels to use. Setting n_levels to 1
      reverts back to a single-scale loss evaluated at the original resolution.
  """
  loss = 0.0
  for level in range(n_levels):
    dr.eval(image, reference, weight)
    loss += loss_fn(image, reference, weight)
    if level < n_levels - 1:
      image = _downsample_image(image)
      reference = _downsample_image(reference)
      if weight is not None:
        weight = _downsample_image(weight)
  return loss
