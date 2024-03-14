import numpy as np
from scipy.optimize import minimize


def primal_dual_splitting(G, F, lambda_param, w, max_iter=1000, tol=1e-6):
    """
    Solve the optimization problem using Primal-Dual Splitting.

    min_H ||G - HF||_2^2 + lambda * sum_ij w_ij |H_ij|

    Parameters:
    G : ndarray
        The matrix G (known image vector).
    F : ndarray
        The matrix F (known image vector).
    lambda_param : float
        Regularization parameter.
    w : ndarray
        Weight matrix.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for convergence.

    Returns:
    H : ndarray
        The optimized matrix H.
    """

    # Initialization
    H = np.zeros_like(G)
    m, n = G.shape

    # Iterative update
    for _ in range(max_iter):
        H_old = H.copy()

        # Update H
        H = np.linalg.inv(F.T @ F + lambda_param * w) @ (F.T @ G)

        # Check for convergence
        if np.linalg.norm(H - H_old) < tol:
            break

    return H


# Example usage (with dummy data)
G = np.random.rand(10, 10)  # Replace with actual image data
F = np.random.rand(10, 10)  # Replace with actual image data
lambda_param = 0.1
w = np.ones((10, 10))  # Replace with actual weight matrix

H_optimized = primal_dual_splitting(G, F, lambda_param, w)
H_optimized
