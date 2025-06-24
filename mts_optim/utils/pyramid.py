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

"""Contains an ImagePyramid class that models images as a multiresolution pyramid."""

from __future__ import annotations

from collections.abc import Callable

import drjit as dr  # type: ignore
import mitsuba as mi  # type: ignore
import numpy as np
import numpy.typing as npt

from mts_optim.utils import image_util


def compute_n_levels(
    shape: tuple[int, int, int],
    factor: int,
    min_resolution: int,
) -> int:
    """Computes the number of levels for an image pyramid.

    For a given image shape, computes the number of levels in a multiresolution
    pyramid such that the minimum resolution of the pyramid is at least
    `min_resolution`.

    E.g., for a shape of (128, 64, 3), a factor of 2 and a minimum resolution of
    8, the function will return 4, corresponding to the following pyramid levels:

      level 0: (128, 64, 3),
      level 1: (64, 32, 3),
      level 2: (32, 16, 3),
      level 3: (16, 8, 3),

    Limitation: This function currently requires that the image shape is a power
    of `factor`.

    Args:
      shape: The shape of the image.
      factor: The resolution scale factor between adjacent levels.
      min_resolution: The minimum resolution of the pyramid.

    Returns:
      The number of levels in the pyramid.
    """
    if min_resolution < 1:
        raise ValueError("Parameter min_resolution has to be >= 1.")

    if min_resolution > max(shape[:2]):
        raise ValueError(
            "Parameter min_resolution has to be <= shape. "
            f"Got min_resolution={min_resolution} and shape={shape}."
        )

    if not all(x.is_integer() for x in np.emath.logn(factor, shape[:2])):
        raise ValueError(f"The shape {shape} has to be a power of {factor}.")

    min_size = min(shape[:2])
    n_levels = int(np.floor(np.emath.logn(factor, min_size))) + 1
    n_levels -= int(np.ceil(np.emath.logn(factor, min_resolution)))
    assert min_size / factor ** (n_levels - 1) >= min_resolution
    return n_levels


class ImagePyramid:
    """Represents an image as multiresolution Laplacian pyramid of images.

    This class represents an image as a stack of images of increasing
    resolution. The original image is recovered by successive upscaling + addition
    of the data in a bottom-up manner. This representation is overparameterized,
    with more degrees of freedom than pixels. This is fine, as the multiscale
    hierarchy means the lower resolution levels get optimized more quickly. This
    representation is used as a preconditioner for gradient descent on image data
    (e.g., textures).

    The class supports nearest neighbor and bilinear interpolation, and also
    supports clamping to a valid range, which limits the values during
    optimization to a physical range."
    """

    def __init__(
        self,
        value: npt.ArrayLike,
        *,
        n_levels: int | None = None,
        min_resolution: int | None = None,
        factor: int = 2,
        shape: tuple[int, int, int] | None = None,
        bilinear_interpolation: bool = False,
    ):
        """Constructs image pyramid from constant value or image.

        Args:
          value: A constant value (single scalar or 3D array) or an image for
            initialization.
          n_levels: Number of levels in the pyramid. The user can either specify the
            number of levels, or the minimum resolution of the pyramid. If both are
            specified, an error is raised. If neither is specified, the maximum
            number of levels will be generated, such that the lowest resolution
            level has a side length of 2 pixels (default: None).
          min_resolution: Minimum resolution of the pyramid (default: None).
          factor: Scale factor between adjacent levels (default: 2).
          shape: Shape of the image if `value` is a constant. If `value` is an
            image, this parameter must be unset.
          bilinear_interpolation: Controls if bilinear interpolation is used.
            Setting it to `False` will use nearest neighbor interpolation (default:
            False).
        """

        if not np.log2(factor).is_integer():
            raise ValueError("factor has to be a power of 2.")

        value = np.asarray(value)
        is_image = np.squeeze(value).ndim > 1
        if is_image:
            if shape is not None:
                raise ValueError(
                    "The 'shape' argument needs to be None for an image.")
            shape = value.shape
        elif shape is None:
            raise ValueError("The 'shape' argument needs to be specified.")

        if n_levels is not None:
            if n_levels < 1:
                raise ValueError("Parameter n_levels has to be >= 1.")
            if min_resolution is not None:
                raise ValueError(
                    "Parameters n_levels and min_resolution cannot both be specified."
                )
        else:
            if min_resolution is None:
                # If neither levels nor min resolution is specified, use a default of 2.
                min_resolution = 2
            n_levels = compute_n_levels(shape, factor, min_resolution)

        self.factor = factor
        self.n_levels = n_levels
        self.shape = shape
        self.bilinear_interpolation = bilinear_interpolation

        self._check_valid_shape()
        if is_image:
            self._from_image(value)
        else:
            reversed_pyramid = []
            image_size = np.array(shape[:2])  # Exclude channels.
            for _ in range(1, self.n_levels):
                level_image = dr.zeros(mi.TensorXf, (*image_size, shape[2]))
                reversed_pyramid.append(level_image)
                image_size = image_size // self.factor

            # Insert the constant value at the lowest level.
            value = value.ravel()
            if len(value) > 1 and len(value) != shape[2]:
                raise ValueError(
                    f"Expected initial value to have 1 or {shape[2]} channels, got"
                    f" {len(value)}."
                )
            reversed_pyramid.append(
                mi.TensorXf(
                    np.ones((*image_size, shape[2])) * value[None, None, :]),
            )

            self.pyramid = list(reversed(reversed_pyramid))

    def _from_image(self, image: np.ndarray):
        """Initializes the image pyramid by an existing image (non-differentiable).

        The result of this conversion is expected to match the input image up to
        numerical differences due to floating point arithmetic. This method
        is *not* differentiable.

        Args:
          image: the image to convert to the pyramid representation.
        """
        divisor = self.factor ** (self.n_levels - 1)
        if self.bilinear_interpolation:
            rfilter = mi.scalar_rgb.load_dict({"type": "tent"})
            # The order of dimensions is reversed for resampling.
            bitmap_res = (image.shape[1] // divisor, image.shape[0] // divisor)
            downsampled = mi.Bitmap(image).resample(bitmap_res, rfilter)
            pyramid = [np.atleast_3d(downsampled)]
            result = 0.0
            for _ in range(1, self.n_levels):
                bitmap_res = (bitmap_res[0] * self.factor,
                              bitmap_res[1] * self.factor)
                target_res = (bitmap_res[1], bitmap_res[0])  # Reverse back.
                result = self._upsample(mi.TensorXf(
                    result + pyramid[-1]), target_res)
                downsampled = mi.Bitmap(image).resample(bitmap_res, rfilter)
                pyramid.append(np.atleast_3d(downsampled) - result)
        else:
            # Don't need to rely on Bitmap class if we simply use nearest neighbor.
            downsampled = [image]
            for _ in range(1, self.n_levels):
                level_image = downsampled[-1]
                for _ in range(int(np.log2(self.factor))):
                    # Compute mean: collapse rows, then columns.
                    level_image = level_image[::2, :,
                                              :] + level_image[1::2, :, :]
                    level_image = level_image[:, ::2,
                                              :] + level_image[:, 1::2, :]
                    level_image /= 4
                downsampled.append(level_image)
            pyramid = downsampled[-1:]
            for i in range(1, self.n_levels):
                diff = downsampled[-(i + 1)] - np.repeat(
                    np.repeat(downsampled[-i], self.factor, axis=0),
                    self.factor,
                    axis=1,
                )
                pyramid.append(diff)
        self.pyramid = [mi.TensorXf(i) for i in pyramid]

    def get_image(self, level: int = -1) -> mi.TensorXf:
        """Differentiable reconstruction of the image represented by the pyramid.

        Args:
          level: The level of the Gaussian pyramid to reconstruct by recursively
            upscaling and summing the levels of the Laplacian pyramid. If level is
            -1, the highest resolution level is returned.

        Returns:
          The reconstructed level of the corresponding Gaussian pyramid.
        """
        if level == -1:
            level = self.n_levels - 1
        if not (0 <= level < self.n_levels):
            raise ValueError(f"level has to be in [0, {(self.n_levels - 1)}].")

        divisor = self.factor ** (self.n_levels - 1)
        target_res = (self.shape[0] // divisor, self.shape[1] // divisor)
        result = 0.0
        for pyramid_level in range(level):
            target_res = (target_res[0] * self.factor,
                          target_res[1] * self.factor)
            result = self._upsample(
                result + self.pyramid[pyramid_level], target_res)
        result += self.pyramid[level]
        dr.eval(result)
        assert dr.grad_enabled(self.pyramid[0]) == dr.grad_enabled(result)
        return result

    def get_flat_image(
        self,
        buffer_offsets: list[int],
    ) -> mi.Float | mi.Color3f:
        """Differentiable reconstruction of the image represented by the pyramid.

        Args:
          buffer_offsets: The buffer offsets for each level of the pyramid.
            (n_levels + 1, starting with 0).

        Returns:
          The reconstructed Gaussian pyramid corresponding to the stored Laplacian
          pyramid. The result is stored in a flat buffer where each level is offset
          by the value given by the corresponding entry in `buffer_offsets[-1]` (The
          last entry is just the size of the flat buffer).
        """
        n_levels = len(buffer_offsets) - 1

        n_channels = self.shape[-1]
        if n_channels not in (1, 3):
            raise ValueError(
                f"Unsupported number of channels {n_channels}, expected 1 or 3"
                " channels."
            )
        storage_type = mi.Color3f if n_channels == 3 else mi.Float

        divisor = self.factor ** (self.n_levels - 1)
        target_res = (self.shape[0] // divisor, self.shape[1] // divisor)
        buffer = dr.zeros(storage_type, shape=buffer_offsets[-1])
        result = 0.0

        for i in range(n_levels):
            result = result + self.pyramid[i]
            value = (
                result.array
                if storage_type == mi.Float
                else dr.unravel(storage_type, result.array)
            )
            index = dr.arange(
                mi.UInt32,
                start=buffer_offsets[n_levels - i - 1],
                stop=buffer_offsets[n_levels - i],
            )
            dr.scatter(buffer, value, index)
            dr.eval(result, buffer)
            if i == n_levels - 1:
                break
            target_res = (
                int(target_res[0] * self.factor),
                int(target_res[1] * self.factor),
            )
            result = self._upsample(result, target_res)

        assert dr.grad_enabled(self.pyramid[0]) == dr.grad_enabled(buffer)
        return buffer

    def clamp(
        self,
        min_value: float,
        max_value: float,
        level: int = -1,
        mipmap_clamping: bool = False,
    ) -> None:
        """Clamps this image pyramid to be within the range [min_value, max_value].

        Args:
          min_value: The minimum value to clamp to.
          max_value: The maximum value to clamp to.
          level: The level up to which the pyramid is clamped. If level is -1, all
            the pyramid levels are clamped.
          mipmap_clamping: If True, the clamping is performed on every reconstructed
            level rather than the highest level only.
        """
        if level == -1:
            level = self.n_levels - 1
        if min_value > max_value:
            raise ValueError(
                "min_value has to be smaller or equal to max_value.")
        if not (0 <= level < self.n_levels):
            raise ValueError(f"level has to be in [0, {(self.n_levels - 1)}].")

        # To avoid floating point errors, we add a small epsilon to the clamping
        # range. (Important to avoid negative values when clamping to 0)
        min_value += 2 * dr.epsilon(mi.Float)
        max_value -= 2 * dr.epsilon(mi.Float)

        # The clamping operation is not intended to be differentiable, so we
        # explicitly disable gradient tracking.
        with dr.suspend_grad():
            if not mipmap_clamping:
                # Note: the following only changes the result of a subsequent
                # get_image() if after going through each level (until `level`) of the
                # pyramid, the last level cannot "compensate" for the difference in the
                # previous levels.

                result = dr.zeros(mi.TensorXf, self.pyramid[0].shape)
                prev_diff = dr.zeros(mi.TensorXf, self.pyramid[0].shape)
                for i in range(level + 1):
                    result_unclamped = result + self.pyramid[i] - prev_diff
                    result_clamped = dr.clip(
                        result_unclamped, min_value, max_value)
                    diff = result_clamped - result_unclamped
                    self.pyramid[i] += diff - prev_diff
                    if i == self.n_levels - 1:
                        break
                    shape = self.pyramid[i].shape
                    shape = (shape[0] * self.factor, shape[1]
                             * self.factor, shape[2])
                    result = self._upsample(result + self.pyramid[i], shape)
                    prev_diff = self._upsample(diff, shape)
                    dr.schedule(self.pyramid[i])
            else:
                self._clamp_levels(
                    level, lambda x: dr.clip(x, min_value, max_value))

    def normalize_channels(self, level: int = -1) -> None:
        """Normalizes channels of this image pyramid.

        This is useful for clamping normal maps to unit vectors.

        Args:
          level: The level up to which the pyramid channels are normalized. If level
            is -1, all the pyramid levels are normalized.
        """
        if level == -1:
            level = self.n_levels - 1
        if not (0 <= level < self.n_levels):
            raise ValueError(f"level has to be in [0, {(self.n_levels - 1)}].")

        # The normalization operation is not intended to be differentiable, so we
        # explicitly disable gradient tracking.
        with dr.suspend_grad():
            self._clamp_levels(level, _normalize_channels)

    def _clamp_levels(
        self, max_level: int, clamp_fn: Callable[[mi.TensorXf], mi.TensorXf]
    ) -> None:
        """Clamp each individual reconstructed level applying the given function."""
        self.pyramid[0] = clamp_fn(self.pyramid[0])
        clamped_image = self.pyramid[0]
        for i in range(1, max_level + 1):
            upsampled = self._upsample(clamped_image, self.pyramid[i].shape)
            original_image = upsampled + self.pyramid[i]
            normalized_raw = clamp_fn(original_image)
            diff = normalized_raw - original_image
            self.pyramid[i] += diff
            clamped_image = upsampled + self.pyramid[i]
            dr.schedule(self.pyramid[i])

    def _check_valid_shape(self) -> None:
        divisor = self.factor ** (self.n_levels - 1)
        if (self.shape[0] % divisor != 0) or (self.shape[1] % divisor != 0):
            raise ValueError(
                f"Image shape {self.shape} is not divisible by factor **"
                f" n_levels = {divisor}."
            )

    def _upsample(self, image, target_shape):
        """Performs upsampling to target_shape."""
        if self.bilinear_interpolation:
            return image_util.bilinear_upsample(
                mi.Texture2f(image), target_shape=target_shape
            )
        else:
            return dr.upsample(image, shape=target_shape)


def _normalize_channels(image: mi.TensorXf) -> mi.TensorXf:
    normals_raw = dr.unravel(mi.Normal3f, image.array)
    normals = (normals_raw * 2.0) - 1
    normalized_raw = (dr.normalize(normals) + 1) * 0.5
    return mi.TensorXf(dr.ravel(normalized_raw), shape=image.shape)
