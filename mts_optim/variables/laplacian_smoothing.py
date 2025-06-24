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

"""Defines a Laplacian smoothing variable."""

from __future__ import annotations

import cholespy  # type: ignore
import drjit as dr  # type: ignore
import mitsuba as mi  # type: ignore
import numpy as np
from scipy import sparse

from mts_optim.variables import array


def _make_edges_3d(n_x, n_y):
    """Returns a list of edges for a 3D image.

    Args:
      n_x: int The size of the grid in the x direction.
      n_y: int The size of the grid in the y direction.
    """
    vertices = np.arange(n_x * n_y).reshape((n_x, n_y, 1))
    edges_deep = np.vstack(
        (vertices[:, :, :-1].ravel(), vertices[:, :, 1:].ravel())
    )
    edges_right = np.vstack(
        (vertices[:, :-1].ravel(), vertices[:, 1:].ravel()))
    edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))
    edges = np.hstack((edges_deep, edges_right, edges_down))
    return edges


def _to_graph(n_x, n_y):
    """Creates the adjacency graph for a 2D image."""
    edges = _make_edges_3d(n_x, n_y)
    dtype = int
    n_voxels = n_x * n_y
    weights = np.ones(edges.shape[1], dtype=dtype)
    diag = np.ones(n_voxels, dtype=dtype)

    diag_idx = np.arange(n_voxels)
    i_idx = np.hstack((edges[0], edges[1]))
    j_idx = np.hstack((edges[1], edges[0]))
    graph = sparse.coo_matrix(
        (
            np.hstack((weights, weights, diag)),
            (np.hstack((i_idx, diag_idx)), np.hstack((j_idx, diag_idx))),
        ),
        (n_voxels, n_voxels),
        dtype=dtype,
    )
    return graph


class SolveConjugateGradient(dr.CustomOp):
    """DrJIT custom operator to solve a linear system using a conjugate gradients."""

    def eval(self, solver, u):
        self.solver = solver
        return solver.solve(u)

    def forward(self):
        x = self.solver.solve(self.grad_in("u"))
        self.set_grad_out(x)

    def backward(self):
        x = self.solver.solve(self.grad_out())
        self.set_grad_in("u", x)

    def name(self):
        return "Conjugate Gradient solve"


class ConjugateGradientSolver:

    def __init__(self, lambda_=19.0, n_iter=64):
        self.n_iter = n_iter
        self.lambda_ = lambda_

    def Ax(self, image: mi.TensorXf) -> mi.TensorXf:
        """Multiplies the image by the system matrix A

        This effectively evaluates the 3x3 kernel:
          0 -1  0            0 0 0
        -1  4 -1 * lambda + 0 1 0
          0 -1  0            0 0 0
        """
        x, y = dr.meshgrid(
            dr.arange(mi.Int32, image.shape[1]),
            dr.arange(mi.Int32, image.shape[0]),
            indexing="xy",
        )

        dtype = mi.Color3f if image.shape[-1] == 3 else mi.Float

        data = image.array
        idx = y * image.shape[1] + x
        pixel_value = dr.gather(dtype, data, idx)
        sum_ = pixel_value * 4
        x0 = dr.maximum(x - 1, 0)
        y0 = dr.maximum(y - 1, 0)
        y1 = dr.minimum(y + 1, image.shape[0] - 1)
        x1 = dr.minimum(x + 1, image.shape[1] - 1)
        sum_ -= dr.gather(dtype, data, dr.fma(y, image.shape[1], x0))
        sum_ -= dr.gather(dtype, data, dr.fma(y, image.shape[1], x1))
        sum_ -= dr.gather(dtype, data, dr.fma(y0, image.shape[1], x))
        sum_ -= dr.gather(dtype, data, dr.fma(y1, image.shape[1], x))
        sum_ = sum_ * self.lambda_ + pixel_value
        return mi.TensorXf(dr.ravel(sum_), image.shape)

    def solve(self, differential: mi.TensorXf) -> mi.TensorXf:
        """Solves a linear system using conjugate gradients.

        Args:
          differential: The input differential coordinates.
          lambda_: The weight of the laplacian regularizer. A higher weight implies
            a stronger regularization.
          n_iter: The number of iterations to run the conjugate gradient solver.

        Returns:
          The resulting image.
        """

        # Notation follows:
        # https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm
        b = differential
        x = dr.zeros(mi.TensorXf, differential.shape)
        r = b  # - Ax(x) # only makes sense if init is not 0
        p = r
        r_t_r = dr.sum(dr.square(r))
        for _ in range(self.n_iter):
            A_p = self.Ax(p)
            dr.eval(A_p)
            a = r_t_r / dr.sum(p * A_p)
            x = x + a * p
            r_next = r - a * A_p
            dr.eval(r_next, x)

            r_t_r_next = dr.sum(dr.square(r_next))
            if r_t_r_next < dr.epsilon(mi.Float):
                break
            p = r_next + r_t_r_next / r_t_r * p

            r = r_next
            r_t_r = r_t_r_next
            dr.eval(r, p, r_t_r)
        dr.eval(x)
        return x


class LaplacianSmoothing:

    def __init__(self, image, lambda_=19.0, use_conjugate_gradient=False):
        self.use_conjugate_gradient = use_conjugate_gradient
        if use_conjugate_gradient:
            print("Creating Conjugate Gradient Solver")
            self.solver = ConjugateGradientSolver(lambda_)
            return

        print("Preparing Laplacian Matrix")
        # Create the adjacency graph for the image.
        W = _to_graph(image.shape[0], image.shape[1])
        degree_matrix = W.sum(axis=1).A1  # .A1 converts it to a 1D numpy array
        D = sparse.diags(degree_matrix)  # Create the sparse diagonal matrix D
        L = D - W

        self.n_verts = image.shape[0] * image.shape[1]
        A = sparse.identity(self.n_verts) + lambda_ * L
        A_coo = A.tocoo()

        print("Creating Cholesky Solver")
        self.row = mi.TensorXi(A_coo.row)
        self.col = mi.TensorXi(A_coo.col)
        data = mi.TensorXd(A_coo.data)
        self.solver = cholespy.CholeskySolverF(
            self.n_verts, self.row, self.col, data, cholespy.MatrixType.COO
        )
        self.data = mi.TensorXf(data)

    def to_differential(self, v) -> mi.TensorXf:
        """Convert a tensor to its smooth laplacian latent tensor: u = (I + λL) v.

        This method typically only needs to be called once per texture, to obtain
        the latent variable before optimization.

        Args:
          v: tensor

        Returns ``mitsuba.Float`:
            Differential form of v.
        """
        if self.use_conjugate_gradient:
            u = self.solver.Ax(v)
            dr.eval(u)
            return u

        storage_type = mi.Color3f if v.shape[2] == 3 else mi.Float
        v_in = dr.unravel(storage_type, v.array)
        u = dr.zeros(storage_type, shape=v_in.shape)

        # Manual sparse matrix multiplication A * v
        row_prod = (
            dr.gather(storage_type, v_in.array,
                      self.col.array) * self.data.array
        )
        dr.scatter_reduce(dr.ReduceOp.Add, u.array, row_prod, self.row.array)
        return dr.reshape(mi.TensorXf, u.array, shape=v.shape)

    def from_differential(self, u) -> mi.TensorXf:
        """Convert Smooth Laplacian Latent tensor back to image form: v = (I +λL)⁻¹ u.

        This is done by solving the linear system (I + λL) v = u using the
        previously computed Cholesky factorization.

        This method is typically called at each iteration of the optimization,
        to update the image (tensor) before rendering.
        """
        if self.use_conjugate_gradient:
            v = dr.custom(
                SolveConjugateGradient,
                self.solver,
                u,
            )
            return v

        v = dr.custom(
            mi.ad.largesteps.SolveCholesky,
            self.solver,
            dr.reshape(mi.TensorXf, u, shape=(self.n_verts, u.shape[2])),
        )
        return dr.reshape(mi.TensorXf, v.array, shape=u.shape)


class LaplacianSmoothingVariable(array.ArrayVariable):
    """Models a Laplacian Smoother variable as a modified tensor variable."""

    def __init__(
        self,
        key: str,
        *,
        initial_value: np.ndarray | mi.TensorXf,
        lambda_: float = 19.0,
        use_conjugate_gradient: bool = False,
        normal_clamping: bool = False,
        **kwargs,
    ):
        super().__init__(key=key, **kwargs)
        del kwargs

        if isinstance(initial_value, np.ndarray):
            initial_value = mi.TensorXf(initial_value)

        self.initial_value = initial_value
        self.shape = initial_value.shape
        self.lambda_ = lambda_
        self.laplacian = None
        self.u = None
        self.use_conjugate_gradient = use_conjugate_gradient
        self.normal_clamping = normal_clamping

    def initialize(
        self,
        optimizer: mi.ad.Optimizer,
        parameters: mi.python.util.SceneParameters,
    ):
        # Initialize the solver and the latent tensor
        self.laplacian = LaplacianSmoothing(
            self.initial_value,
            self.lambda_,
            use_conjugate_gradient=self.use_conjugate_gradient,
        )
        self.u = self.laplacian.to_differential(self.initial_value)
        optimizer[self.optimizer_key] = self.u
        optimizer.set_learning_rate(self.get_learning_rates())

    def update(
        self,
        optimizer: mi.ad.Optimizer,
        parameters: mi.python.util.SceneParameters,
        iteration: int,
    ):
        assert self.laplacian is not None
        if self.clamp_range is not None or self.normal_clamping:
            with dr.suspend_grad():
                u = self.laplacian.from_differential(
                    optimizer[self.optimizer_key])
                if self.normal_clamping:
                    normals = dr.unravel(mi.Normal3f, u.array)
                    normalized = (dr.normalize((normals * 2.0) - 1) + 1) * 0.5
                    u = mi.TensorXf(dr.ravel(normalized), shape=u.shape)
                else:
                    u = dr.clamp(
                        u,
                        self.clamp_range[0],
                        self.clamp_range[1],
                    )
                optimizer[self.optimizer_key] = self.laplacian.to_differential(
                    u)
            dr.enable_grad(optimizer[self.optimizer_key])

    def get_value(self, optimizer: mi.ad.Optimizer) -> mi.TensorXf:
        if self.laplacian is None:
            raise ValueError(
                f"Laplacian variable {self.key} has not been initialized yet!"
            )
        u = optimizer[self.optimizer_key]
        v = self.laplacian.from_differential(u)
        return v
