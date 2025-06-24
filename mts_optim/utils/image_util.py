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

"""This module contains self-contained utility functions."""

from __future__ import annotations

import drjit as dr  # type: ignore
import mitsuba as mi  # type: ignore
import numpy as np

from mts_optim.utils import mitsuba_util

_BOXFILTER = mi.scalar_rgb.load_dict({'type': 'box'})


def resize_to_width(
    image: np.ndarray | mi.Bitmap | mi.TensorXf,
    desired_width: int,
) -> np.ndarray | mi.Bitmap | mi.TensorXf:
    """Resize an image to a desired width while preserving the dimensionality."""
    input_type = type(image)
    input_shape = None
    if isinstance(image, np.ndarray) or isinstance(image, mi.TensorXf):
        input_shape = image.shape
        image = mi.Bitmap(image)

    width = image.width()
    if width == desired_width:
        resampled = image
    else:
        new_height = int((desired_width / width) * image.height())
        resampled = image.resample((desired_width, new_height), _BOXFILTER)

    if input_type == np.ndarray:
        result = np.array(resampled)
        if len(input_shape) == 3 and input_shape[-1] == 1:
            result = np.atleast_3d(result)
    elif input_type == mi.TensorXf:
        result = mi.TensorXf(resampled)
        if len(input_shape) == 2 and len(result.shape) == 3:
            result = result[:, :, 0]
    else:
        result = mi.Bitmap(resampled)
    return result


def tonemap(
    image: np.ndarray | mi.Bitmap | mi.TensorXf,
) -> mi.Bitmap:
    """Tonemaps the given image to the sRGB color space.

    Args:
      image: The image to tonemap. Note: the image must contain 32-bit data!

    Returns:
      A Mitsuba Bitmap in sRGB color space and 8-bit precision.
    """
    if isinstance(image, np.ndarray) and not issubclass(
        image.dtype.type, np.floating
    ):
        raise ValueError(
            'Tone mapping a non-float image does not really make sense.'
        )
    bitmap = mi.Bitmap(image)
    return bitmap.convert(
        mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True
    )


def clamp_multiscale_level(
    resolution: int, min_resolution: int, n_levels: int, factor: int
) -> int:
    """Ensures a minimum texture resolution for multiscale representations.

    Args:
      resolution: The target resolution of the output in pixels.
      min_resolution: The desired minimum resolution in pixels.
      n_levels: The number of multiscale or upsampling levels.
      factor: The scaling factor between subsequent levels.

    Returns:
      The number of levels, adjusted such that the minimum achieved resolution
      is at least the provided minimum.
    """
    if min_resolution > resolution:
        raise ValueError(
            f'The min_resolution (provided: {min_resolution}) must be smaller than'
            f' the resolution (provided: {resolution}).'
        )

    current = resolution
    level = 1
    while level < n_levels:
        current = current // factor
        if current > min_resolution:
            level += 1
        else:
            break
    assert resolution // (factor ** (level - 1)) >= min_resolution

    return level


def apply_color_transform(
    image: mi.TensorXf, transform: mi.Transform4f | mi.Matrix3f | mi.Color3f
) -> mi.TensorXf:
    """Returns the color-transformed input RGB image.

    Args:
      image: 3-channel RGB image encoded as a Mitsuba tensor.
      transform: either a full 4x4 color transform matrix or a 3-vector that is
        implicitly treated as a diagonal color transformation.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f'Only 3-channel images are supported but got {image.shape=}!'
        )

    if isinstance(transform, mi.Color3f):
        return image * mi.TensorXf(dr.ravel(transform), shape=(1, 1, 3))

    return mi.TensorXf(
        dr.ravel(transform @ dr.unravel(mi.Color3f, image.array)),
        shape=image.shape,
    )


def get_nonzero_region(
    image: mi.TensorXf | np.array, padding: int = 0
) -> mitsuba_util.CropWindow | None:
    """Get the bounding box of the region where the mask is non-zero.

    Args:
      image: The input image.
      padding: The padding size of the bounding box.

    Returns:
      The bounding box of the non-zero region in the image as a crop window.
      A value of `None` is returned if there are no valid mask pixels.
    """

    y, x = dr.meshgrid(
        dr.arange(mi.Int32, image.shape[0]),
        dr.arange(mi.Int32, image.shape[1]),
        indexing='ij',
    )
    valid = mi.TensorXf(np.max(image, axis=-1)).array > 0.0
    if not dr.any(valid):
        return None
    max_size = np.iinfo(np.int32).max
    x_min = dr.min(dr.select(valid, x, max_size)) - padding
    x_max = dr.max(dr.select(valid, x, -max_size)) + padding
    y_min = dr.min(dr.select(valid, y, max_size)) - padding
    y_max = dr.max(dr.select(valid, y, -max_size)) + padding
    x_min = dr.maximum(x_min, 0)
    x_max = dr.minimum(x_max, image.shape[1])
    y_min = dr.maximum(y_min, 0)
    y_max = dr.minimum(y_max, image.shape[0])

    return mitsuba_util.CropWindow(
        offset=(x_min[0], y_min[0]),
        size=(x_max[0] - x_min[0], y_max[0] - y_min[0]),
    )


def bilinear_upsample(
    texture: mi.Texture2f, target_shape: tuple[int, int] | tuple[int, int, int]
) -> mi.TensorXf:
    """Bilinearly upsamples a texture.

    Args:
      texture: The input texture.
      target_shape: The desired shape of the output texture.

    Returns:
      The upsampled texture as a Mitsuba Tensor.
    """
    shape_ = list(target_shape) + list(texture.shape[len(target_shape):])

    value_type = type(texture.value())
    dim = len(texture.shape) - 1

    if texture.shape[dim] != shape_[dim]:
        raise TypeError(
            "upsample(): channel counts doesn't match input texture!")

    # Create the query coordinates
    coords = list(
        dr.meshgrid(
            *[
                dr.linspace(value_type, 0.0, 1.0, shape_[i], endpoint=False)
                for i in range(dim)
            ],
            indexing='ij',
        )
    )

    # Offset coordinates by half a voxel to hit the center of the new voxels
    for i in range(dim):
        coords[i] += 0.5 / shape_[i]

    # Reverse coordinates order according to dr.Texture convention
    coords.reverse()

    # Evaluate the texture at all voxel coordinates with interpolation
    values = texture.eval(coords)

    # Concatenate output values to a flatten buffer
    channels = len(values)
    flat_size = dr.width(values[0])
    index = dr.arange(dr.uint32_array_t(value_type), flat_size)
    data = dr.zeros(value_type, flat_size * channels)
    for c in range(channels):
        dr.scatter(data, values[c], channels * index + c)

    return mi.TensorXf(
        data, shape=(target_shape[0], target_shape[1], texture.shape[dim])
    )
