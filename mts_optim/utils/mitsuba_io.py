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

"""Module that provides IO functions to write out data."""
from __future__ import annotations

import os
import time
from tqdm import tqdm
from typing import List, Union

import mitsuba as mi  # type: ignore

from pathlib import Path


def _approximate_atomic_write(filename: Path | str, bytestream: bytes):
  """Approximately atomic file write, to avoid race conditions on Flume."""

  filename = Path(filename)

  filename_tmp = filename.with_name(f"{filename.name}.{os.getpid()}.{time.time()}.tmp")
  with open(filename_tmp, 'wb') as f:
    f.write(bytestream)

  # For windows
  if filename.exists():
    filename.unlink()

  filename_tmp.rename(filename)


def read_bitmap(filename: str | Path) -> mi.Bitmap:
  """Reads out a Mitsuba Bitmap image from the specified filepath.

  This function is a thin wrapper around Mitsuba's Bitmap class to read image
  files from custom filesystems.

  Args:
    filename: The path to the file to read the image from.

  Returns:
    A Mitsuba Bitmap with the loaded image.
  """
  stream = mi.MemoryStream()
  with open(filename, 'rb') as f:
    stream.write(f.read())
  stream.seek(0)
  return mi.Bitmap(stream)


def write_bitmap(image: mi.Bitmap, filename: str | Path):
  """Writes out a Mitsuba Bitmap to a specified filepath.

  This function is a thin wrapper around the Mitsuba Bitmap's `write` function.

  This extends Mitsuba's Bitmap writing by support for:
    - Writing images to custom filesystems.
    - Approximately atomic writes, which might be needed when running jobs on servers. 
      Internally, this is done by first writing to a unique temporary
      file and renaming the file upon completion.

  Args:
    image: The Bitmap object to write.
    filename: The path to the file to write the Bitmap to.
  """
  filetype_map = {
      '.jpg': mi.Bitmap.FileFormat.JPEG,
      '.jpeg': mi.Bitmap.FileFormat.JPEG,
      '.exr': mi.Bitmap.FileFormat.OpenEXR,
      '.png': mi.Bitmap.FileFormat.PNG,
      '.pfm': mi.Bitmap.FileFormat.PFM,
      '.ppm': mi.Bitmap.FileFormat.PPM,
      '.rgbe': mi.Bitmap.FileFormat.RGBE,
      '.hdr': mi.Bitmap.FileFormat.RGBE,
  }
  ext = os.path.splitext(filename)[-1]
  if ext not in filetype_map:
    raise ValueError('Unsupported file type: ' + ext)
  filetype = filetype_map[ext]
  stream = mi.MemoryStream()
  image.write(stream, filetype)
  _approximate_atomic_write(filename, stream.raw_buffer())


def write_mesh_ply(mesh: mi.Mesh, filename: str | Path):
  """Writes the given Mitsuba mesh to a binary PLY file.

  This extends Mitsuba's PLY Mesh writing by support for:
    - Writing meshes to custom filesystems.
    - Approximately atomic writes, which might be needed when running jobs on servers. 
      Internally, this is done by first writing to a unique temporary
      file and renaming the file upon completion

  Args:
    mesh: The mesh to write.
    filename: The path to the file to write the mesh to.
  """
  stream = mi.MemoryStream()
  mesh.write_ply(stream)
  _approximate_atomic_write(filename, stream.raw_buffer())


def read_images(
    filenames: List[Union[str, Path]],
    message: str | None = None,
) -> List[mi.Bitmap]:
  """Reads a list of images sequentially.

  Args:
    filenames: A list of filenames (strings or pathlib.Path objects).
    message: Optional status string shown by tqdm on the progress bar.

  Returns:
    List of Mitsuba Bitmap images.
  """

  images = []
  progress_bar = tqdm(filenames, desc=message)

  for filename in progress_bar:
    try:
      images.append(mi.Bitmap(str(filename)))
    except Exception as e:
      print(f"Error reading image {filename}: {e}")
      # Handle the error as needed, e.g., skip the image or raise an exception
      pass

  return images