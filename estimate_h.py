"""estimate_h.py"""

# pylint: disable=invalid-name

import os
import math
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as csp


def prox_l122(Y: cp.ndarray, gamma: float, N: int) -> cp.ndarray:
    """
    L1,2ノルムの2乗のprox
    """
    factor = (2 * gamma) / (1 + 2 * gamma * N)
    l1_norms = cp.sum(cp.absolute(Y), axis=1)
    X = cp.sign(Y) * cp.maximum(cp.absolute(Y) - factor * l1_norms[:, None], 0)
    return X


def fista(
    F: np.ndarray,
    G: np.ndarray,
    M: int,
    lmd: float,
    max_iter: int = 150,
) -> csp.csr_matrix:
    """
    FISTA
    """
    N = F.shape[0]
    L = N
    gamma = 1.0 / (L * 3)

    t_memo = np.ones(max_iter + 1, dtype=np.float32)
    for i in range(1, max_iter + 1):
        t_memo[i] = (1 + math.sqrt(1 + 4 * t_memo[i - 1] ** 2)) / 2

    H = cp.zeros((M, N), dtype=cp.float32)
    H_old = cp.zeros((M, N), dtype=cp.float32)
    Y = cp.zeros((M, N), dtype=cp.float32)
    F_gpu = cp.asarray(F)
    G_gpu = cp.asarray(G)
    for i in range(max_iter):
        H_old = H.copy()
        H = prox_l122(Y - gamma * (Y @ F_gpu - G_gpu) @ F_gpu.T, gamma * lmd, N)
        Y = H + ((t_memo[i] - 1) / t_memo[i + 1]) * (H - H_old)
        if i % 10 == 0:
            print(f"Iteration {i+1}/{max_iter}")

    return csp.csr_matrix(H)


if __name__ == "__main__":
    cap_dates = {128: "241114", 256: "241205"}
    n = 128
    LAMBDA = 100
    RATIO = 0.05
    DATA_PATH = "../data"
    CAP_DATE = cap_dates[n]
    EXP_DATE = "241230"
    DIRECTORY = f"{DATA_PATH}/{EXP_DATE}"
    SETTING = f"{n}_p-{int(100*RATIO)}_lmd-{LAMBDA}"

    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)
    if not os.path.exists(DIRECTORY + "/systemMatrix"):
        os.makedirs(DIRECTORY + "/systemMatrix")

    F_hat = np.load(f"{DATA_PATH}/capture_{CAP_DATE}/F_hat.npy").astype(np.int8)
    G_hat = np.load(f"{DATA_PATH}/capture_{CAP_DATE}/G_hat.npy").astype(np.float32)
    print(f"F_hat shape: {F_hat.shape}, dtype: {F_hat.dtype}")
    print(f"G_hat shape: {G_hat.shape}, dtype: {G_hat.dtype}")

    H_csp = fista(F_hat, G_hat, G_hat.shape[0], LAMBDA)

    print(f"shape: {H_csp.shape}")
    print(f"nnz ratio: {H_csp.nnz}({H_csp.nnz / (H_csp.shape[0] * H_csp.shape[1]) * 100:.2f}%)")
    H_np = {
        "data": cp.asnumpy(H_csp.data),
        "indices": cp.asnumpy(H_csp.indices),
        "indptr": cp.asnumpy(H_csp.indptr),
        "shape": H_csp.shape,
    }
    np.savez(f"{DIRECTORY}/systemMatrix/H_matrix_{SETTING}.npz", **H_np)
    print(f"Saved {DIRECTORY}/systemMatrix/H_matrix_{SETTING}.npz")
