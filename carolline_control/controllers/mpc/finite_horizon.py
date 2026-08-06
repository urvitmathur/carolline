"""Finite-horizon linear MPC (condensed QP + projected gradient for input bounds)."""

from __future__ import annotations

import numpy as np


def discretize_euler(Ac: np.ndarray, Bc: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    n = Ac.shape[0]
    return np.eye(n) + dt * Ac, dt * Bc


def _prediction_matrices(Ad: np.ndarray, Bd: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Stack x_1..x_N: x_stack = M @ x0 + Phi @ U."""
    nx, nu = Ad.shape[0], Bd.shape[1]
    M = np.zeros((nx * horizon, nx))
    Phi = np.zeros((nx * horizon, nu * horizon))
    for k in range(horizon):
        row = slice(k * nx, (k + 1) * nx)
        Ak = np.linalg.matrix_power(Ad, k + 1)
        M[row, :] = Ak
        for j in range(k + 1):
            col = slice(j * nu, (j + 1) * nu)
            Aij = np.linalg.matrix_power(Ad, k - j)
            Phi[row, col] = Aij @ Bd
    return M, Phi


def _cost_matrices(
    Q: np.ndarray,
    R: np.ndarray,
    Qf: np.ndarray,
    horizon: int,
    Phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    nx = Q.shape[0]
    blocks = [Q] * (horizon - 1) + [Qf] if horizon > 0 else [Qf]
    Q_stack = np.block([[blocks[i] if i == j else np.zeros((nx, nx)) for j in range(len(blocks))] for i in range(len(blocks))])
    R_stack = np.kron(np.eye(horizon), R)
    H = Phi.T @ Q_stack @ Phi + R_stack
    H = 0.5 * (H + H.T) + 1e-9 * np.eye(H.shape[0])
    return H, Q_stack


class LinearMpcSolver:
    """Precomputed condensed MPC matrices for fast receding-horizon solves."""

    def __init__(
        self,
        Ad: np.ndarray,
        Bd: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        Qf: np.ndarray,
        horizon: int,
        *,
        u_min: np.ndarray | None = None,
        u_max: np.ndarray | None = None,
        pg_iters: int = 40,
    ) -> None:
        self._nu = Bd.shape[1]
        self._horizon = horizon
        self._pg_iters = pg_iters
        M, Phi = _prediction_matrices(Ad, Bd, horizon)
        H, Q_stack = _cost_matrices(Q, R, Qf, horizon, Phi)
        self._PhiTQ = Phi.T @ Q_stack
        self._MQ = self._PhiTQ @ M
        self._H = H
        self._H_unconstrained = np.linalg.solve(H, np.eye(H.shape[0]))
        self._has_bounds = u_min is not None and u_max is not None
        if self._has_bounds:
            self._lo = np.tile(u_min, horizon)
            self._hi = np.tile(u_max, horizon)
            self._pg_step = 1.0 / (np.linalg.norm(H, ord=2) + 1e-6)

    def solve(self, x_err: np.ndarray) -> np.ndarray:
        f = self._MQ @ x_err
        if not self._has_bounds:
            U = -self._H_unconstrained @ f
        else:
            U = -self._H_unconstrained @ f
            for _ in range(self._pg_iters):
                U = U - self._pg_step * (self._H @ U + f)
                U = np.clip(U, self._lo, self._hi)
        return U[: self._nu].copy()


def solve_linear_mpc(
    Ad: np.ndarray,
    Bd: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    Qf: np.ndarray,
    horizon: int,
    x_err: np.ndarray,
    *,
    u_min: np.ndarray | None = None,
    u_max: np.ndarray | None = None,
    pg_iters: int = 40,
) -> np.ndarray:
    """First control u0 for regulation (x_err -> 0) with optional input bounds."""
    solver = LinearMpcSolver(
        Ad, Bd, Q, R, Qf, horizon, u_min=u_min, u_max=u_max, pg_iters=pg_iters
    )
    return solver.solve(x_err)
