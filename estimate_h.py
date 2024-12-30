import os
import math
import multiprocessing
from typing import List
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as csp


def initialize_gpu(gpu_id):
    cp.cuda.Device(gpu_id).use()


def prox_l122(Y: cp.ndarray, gamma: float, N: int) -> cp.ndarray:
    factor = (2 * gamma) / (1 + 2 * gamma * N)
    l1_norms = cp.sum(cp.absolute(Y), axis=1)
    X = cp.sign(Y) * cp.maximum(cp.absolute(Y) - factor * l1_norms[:, None], 0)
    return X


def fista_chunk(
    gpu_id: int,
    F: cp.ndarray,
    G_chunk: cp.ndarray,
    lmd: float,
    gamma: float,
    max_iter: int,
    t_memo: cp.ndarray,
) -> csp.csr_matrix:
    """
    各チャンクのFISTA処理
    """
    initialize_gpu(gpu_id)
    N = F.shape[0]
    rows = G_chunk.shape[0]
    H_chunk = cp.zeros((rows, N), dtype=cp.float32)
    H_chunk_old = cp.zeros((rows, N), dtype=cp.float32)
    Y_chunk = cp.zeros((rows, N), dtype=cp.float32)
    G_chunk_gpu = cp.asarray(G_chunk)

    for i in range(max_iter):
        H_chunk_old = H_chunk.copy()
        H_chunk = prox_l122(Y_chunk - gamma * (Y_chunk @ F - G_chunk_gpu) @ F.T, gamma * lmd, N)
        Y_chunk = H_chunk + ((t_memo[i] - 1) / t_memo[i + 1]) * (H_chunk - H_chunk_old)
        print(f"Process {gpu_id+1} | Iteration {i+1}/{max_iter}")

    return csp.csr_matrix(H_chunk)


def fista_parallel(
    F: cp.ndarray,
    G: np.ndarray,
    M: int,
    lmd: float,
    max_iter: int = 150,
) -> csp.csr_matrix:
    """
    FISTAの並列バージョン
    """
    gpu_ids = [0, 1]
    num_gpus = len(gpu_ids)
    num_processes = 4
    N = F.shape[0]
    L = N
    gamma = 1.0 / (L * 3)

    t_memo = cp.ones(max_iter + 1, dtype=cp.float32)
    for i in range(1, max_iter + 1):
        t_memo[i] = (1 + math.sqrt(1 + 4 * t_memo[i - 1] ** 2)) / 2

    chunk_size = math.ceil(M / num_processes)
    chunks: List[csp.csr_matrix] = []
    futures = []

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for c in range(num_processes):
            gpu_id = gpu_ids[c % num_gpus]
            start = c * chunk_size
            end = min((c + 1) * chunk_size, M)
            if start >= M:
                break
            G_chunk = G[start:end, :]

            print(f"Process {c+1}/{num_processes} on GPU {gpu_id} | rows: {G_chunk.shape[0]}")
            futures.append(executor.submit(fista_chunk, gpu_id, F, G_chunk, lmd, gamma, max_iter, t_memo))

        for future in futures:
            chunks.append(future.result())

    H_csp = csp.vstack(chunks).tocsr()
    return H_csp


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

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
    F_hat = cp.load(f"{DATA_PATH}/capture_{CAP_DATE}/F_hat.npy")
    G_hat = np.load(f"{DATA_PATH}/capture_{CAP_DATE}/G_hat.npy")

    H = fista_parallel(F_hat, G_hat, G_hat.shape[0], LAMBDA)

    print(f"shape: {H.shape}, nnz: {H.nnz}({H.nnz / H.shape[0] / H.shape[1] * 100:.2f}%)")
    H_np = {
        "data": cp.asnumpy(H.data),
        "indices": cp.asnumpy(H.indices),
        "indptr": cp.asnumpy(H.indptr),
        "shape": H.shape,
    }
    np.savez(f"{DIRECTORY}/systemMatrix/H_matrix_{SETTING}.npz", **H_np)
    print(f"Saved {DIRECTORY}/systemMatrix/H_matrix_{SETTING}.npz")
