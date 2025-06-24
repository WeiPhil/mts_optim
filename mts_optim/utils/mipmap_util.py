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


def get_mip_levels(
    si: mi.SurfaceInteraction3f,
    max_mip_levels: mi.UInt,
    n_mip_levels: mi.UInt32,
    mip_bias: mi.Float = 0.0,
) -> tuple[mi.UInt32, mi.UInt32, mi.Float]:
  """Computes the mipmap levels and weight for a given surface interaction.

  Downsampling factor is assumed to be 2.

  Args:
    si: The surface interaction.
    max_mip_levels: The maximum number of mip levels.
    n_mip_levels: The number of available mip levels.
    mip_bias: The bias to add to the mip level.

  Returns:
    A tuple of two consecutive mip levels and the interpolation weight.
  """
  width = dr.maximum(dr.max(dr.abs(si.duv_dx)), dr.max(dr.abs(si.duv_dy)))
  width = dr.maximum(width, dr.epsilon(mi.Float))
  mip_level = dr.clip(
      mi.Float(max_mip_levels) + dr.log2(width) + mip_bias,
      0.0,
      n_mip_levels - 1,
  )
  int_mip_level = dr.floor(mip_level)
  mip_weight = mip_level - int_mip_level
  # Make sure to utilize the int_mip_level as it might be -1 in which only the
  # highest resolution level is used.
  mip_level_1 = mi.UInt(int_mip_level)
  mip_level_2 = dr.minimum(mip_level_1 + 1, n_mip_levels - 1)
  return mip_level_1, mip_level_2, mip_weight


def create_flat_mip_buffer(
    bitmap_data: mi.TensorXf,
    factor: int = 2,
    max_levels: int | None = None,
    min_resolution: int = 2,
) -> tuple[mi.Float | mi.Color3f, list[int]]:
  """Creates a flat mip buffer from a bitmap and a downsampling factor.

  Args:
    bitmap_data: A list of tensors, each representing a mip level.
    factor: The downsampling factor.
    max_levels: The maximum number of mip levels to generate.
    min_resolution: The minimum resolution of the mip levels.

  Returns:
    A flattened (concatenated) buffer and a list of offsets for each mip level.
  """
  min_resolution = max(min_resolution, factor)
  rfilter = mi.scalar_rgb.load_dict({"type": "gaussian"})
  mip_data = [bitmap_data]
  # Swapping width/height to match Bitmap shape convention.
  target_size = mi.UInt(bitmap_data.shape[1::-1])

  # Third condition required because mi.Bitmap requires at least 2x2 textures.
  while (
      all(target_size % factor == 0)
      and min(target_size) > min_resolution
      and (max_levels is None or len(mip_data) < max_levels)
  ):
    target_size = target_size // factor
    bitmap_mip_data = mi.TensorXf(
        mi.Bitmap(bitmap_data).resample(target_size, rfilter)
    )
    mip_data.append(bitmap_mip_data)

  if mip_data[0].shape[-1] not in {1, 3}:
    raise ValueError(
        f"Unsupported number of channels {mip_data[0].shape[-1]}, expected 1 or"
        " 3 channels."
    )

  storage_type = mi.Color3f if mip_data[0].shape[-1] == 3 else mi.Float

  buffer_offsets = [0]
  for data in mip_data:
    next_buffer_size = buffer_offsets[-1] + data.shape[0] * data.shape[1]
    buffer_offsets.append(next_buffer_size)

  buffer_offsets = mi.UInt32(buffer_offsets)

  buffer = dr.zeros(storage_type, shape=buffer_offsets[-1])
  for level, mip_level_data in enumerate(mip_data):
    buffer_start = buffer_offsets[level]
    buffer_end = buffer_offsets[level + 1]
    flat_mip_data = (
        mip_level_data.array
        if storage_type == mi.Float
        else dr.unravel(storage_type, mip_level_data.array)
    )
    dr.scatter(
        buffer,
        flat_mip_data,
        dr.arange(mi.UInt32, start=buffer_start, stop=buffer_end),
    )

  return buffer, buffer_offsets


def mip_tensor_from_flat_buffer(
    mip_level: int,
    buffer: mi.Float,
    buffer_offsets: list[int],
    base_shape: mi.ScalarVector3u,
    factor: int,
) -> mi.TensorXf:
  """Retrieves a given mip level from a flat buffer as a TensorXf, buffer offsets and the storage type.

  Args:
    mip_level: The mip level to extract.
    buffer: The flat buffer.
    buffer_offsets: A list of offsets for each mip level.
    base_shape: The shape of the highest resolution mip level.
    factor: The downsampling factor of the mipmap.

  Returns:
    A tensor representing the queried mip level.
  """
  base_shape = tuple(base_shape)
  mip_shape = (
      base_shape[0] // factor**mip_level,
      base_shape[1] // factor**mip_level,
      base_shape[2],
  )
  if (
      mip_shape[0] * mip_shape[1]
      != buffer_offsets[mip_level + 1] - buffer_offsets[mip_level]
  ):
    raise ValueError(
        f"The buffer offset for mip level {mip_level} is"
        f" {buffer_offsets[mip_level + 1] - buffer_offsets[mip_level]} but the"
        f" shape is {mip_shape[0]} x {mip_shape[1]}!"
    )

  if base_shape[-1] not in {1, 3}:
    raise ValueError(
        f"Unsupported number of channels {base_shape[-1]}, expected 1 or"
        " 3 channels."
    )

  storage_type = mi.Color3f if base_shape[-1] == 3 else mi.Float

  mip_data = dr.gather(
      storage_type,
      buffer,
      dr.arange(
          mi.UInt32,
          start=buffer_offsets[mip_level],
          stop=buffer_offsets[mip_level + 1],
      ),
  )

  if storage_type != mi.Float:
    mip_data = dr.ravel(mip_data)
  return mi.TensorXf(mip_data, shape=mip_shape)
