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

"""Mitsuba-related utility functions.

This file contains utility functions for interacting with Mitsuba.
This bundles functionality that is unrelated to the main inverse rendering
pipeline, but too specific for vr/perception/common/mitsuba.
"""

from __future__ import annotations

import dataclasses

import gin
import mitsuba as mi  # type: ignore
import numpy as np


@gin.configurable
@dataclasses.dataclass
class CropWindow:
  """Represents a crop window in an image.

  size: the size of the crop window
  offset: offset of the crop window in the image.
  """

  offset: tuple[int, int]
  size: tuple[int, int]

  def __post_init__(self):
    if self.offset[0] < 0 or self.offset[1] < 0:
      raise ValueError('Invalid crop offset!')
    if self.size[0] < 0 or self.size[1] < 0:
      raise ValueError('Invalid crop size!')


def set_sensor_resolution(
    sensor: mi.Sensor,
    resolution: tuple[int, int],
    crop_window: CropWindow | None = None,
) -> None:
  """Sets the resolution of a Mitsuba sensor.

  Args:
    sensor: The sensor to be resized.
    resolution: The resolution as (width, height).
    crop_window: The crop window to be applied to the sensor.
  """
  params = mi.traverse(sensor)
  params['film.size'] = resolution
  if crop_window is not None:
    params['film.crop_offset'] = [crop_window.x, crop_window.y]
    params['film.crop_size'] = [crop_window.width, crop_window.height]
  params.update()


def copy_reconstruction_filter(
    reconstruction_filter: mi.ReconstructionFilter,
) -> mi.ReconstructionFilter:
  """Creates a deep copy of a Mitsuba reconstruction filter.

  Args:
    reconstruction_filter: The reconstruction filter to be copied.

  Returns:
    A new Mitsuba reconstruction filter instance with the same parameters as the
    input reconstruction filter.
  """

  class_name = reconstruction_filter.class_().name()

  match class_name:
    case 'BoxFilter':
      return mi.load_dict({'type': 'box'})
    case 'GaussianFilter':
      return mi.load_dict(
          {'type': 'gaussian', 'stddev': reconstruction_filter.radius() / 4}
      )
    case _:
      raise ValueError(
          'Expected a BoxFilter or GaussianFilter, but got %s' % class_name
      )


def copy_film(film: mi.Film) -> mi.Film:
  """Creates a deep copy a Mitsuba film.

  Args:
    film: The film to be copied.

  Returns:
    A new Mitsuba film instance with the same parameters as the input film.
  """

  class_name = film.class_().name()
  if class_name != 'HDRFilm':
    raise ValueError('Expected a HDRFilm, but got %s' % class_name)

  if film.base_channels_count() != 3:
    raise ValueError(
        'Expected a HDRFilm with 3 channels, but got %s'
        % film.base_channels_count()
    )
  film_params = mi.traverse(film)
  return mi.load_dict({
      'type': 'hdrfilm',
      'height': film_params['size'][1],
      'width': film_params['size'][0],
      'rfilter': copy_reconstruction_filter(film.rfilter()),
      'sample_border': film.sample_border(),
      'pixel_format': 'rgb',
      'crop_offset_x': film_params['crop_offset'][0],
      'crop_offset_y': film_params['crop_offset'][1],
      'crop_width': film_params['crop_size'][0],
      'crop_height': film_params['crop_size'][1],
  })


def copy_sensor(sensor: mi.Sensor) -> mi.Sensor:
  """Copies a Mitsuba sensor.

  This creates a deep copy of an input sensor. Most importanty, it also
  creates a new film instance. This makes it save to copy a sensor and then
  set a different resolution on the new sensor.

  Args:
    sensor: The sensor to be copied.

  Returns:
    A new Mitsuba sensor instance with the same parameters as the input sensor.
  """

  class_name = sensor.class_().name()
  if class_name not in ['PerspectiveCamera', 'ThinLensCamera']:
    raise ValueError(
        'Expected a PerspectiveCamera or ThinLensCamera, but got %s'
        % class_name
    )

  sensor_params = mi.traverse(sensor)

  camera_dict = {
      'fov': float(sensor_params['x_fov'][0]),
      'near_clip': sensor_params['near_clip'],
      'far_clip': sensor_params['far_clip'],
      'film': copy_film(sensor.film()),
      'to_world': mi.ScalarTransform4f(
          mi.ScalarMatrix4f(np.array(sensor_params['to_world'].matrix)[..., 0])
      ),
  }

  if class_name == 'PerspectiveCamera':
    camera_dict |= {
        'type': 'perspective',
        'principal_point_offset_x': float(
            sensor_params['principal_point_offset_x'][0]
        ),
        'principal_point_offset_y': float(
            sensor_params['principal_point_offset_y'][0]
        ),
    }
  else:
    camera_dict |= {
        'type': 'thinlens',
        'focus_distance': float(sensor_params['focus_distance'][0]),
        'aperture_radius': float(sensor_params['aperture_radius'][0]),
    }

  return mi.load_dict(camera_dict)
