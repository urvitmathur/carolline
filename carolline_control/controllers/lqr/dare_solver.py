"""Discrete-time LQR via iterative DARE (numpy only, no SciPy dependency)."""

from __future__ import annotations

import numpy as np


def discretize_euler(Ac: np.ndarray, Bc: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Forward-Euler discretization x[k+1] = Ad x[k] + Bd u[k]."""
    n = Ac.shape[0]
    return np.eye(n) + dt * Ac, dt * Bc


def solve_dare(
    Ad: np.ndarray,
    Bd: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    *,
    max_iter: int = 800,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (K, P) for u = -K x minimizing sum x'Qx + u'Ru."""
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    P = Q.copy()
    K = np.zeros((Bd.shape[1], Ad.shape[0]))
    for _ in range(max_iter):
        S = R + Bd.T @ P @ Bd
        K = np.linalg.solve(S, Bd.T @ P @ Ad)
        P_new = Q + Ad.T @ P @ (Ad - Bd @ K)
        if float(np.max(np.abs(P_new - P))) < tol:
            P = P_new
            break
        P = P_new
    S = R + Bd.T @ P @ Bd
    K = np.linalg.solve(S, Bd.T @ P @ Ad)
    return K, P


def check_closed_loop_stable(Ad: np.ndarray, Bd: np.ndarray, K: np.ndarray) -> bool:
    """True when spectral radius of Ad - Bd K is strictly inside unit disk."""
    A_cl = Ad - Bd @ K
    eigvals = np.linalg.eigvals(A_cl)
    return bool(np.max(np.abs(eigvals)) < 1.0)
