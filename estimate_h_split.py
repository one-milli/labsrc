"""estimate_h_split.py"""

# pylint: disable=invalid-name
# pylint: disable=line-too-long

import os
import math
import time
import multiprocessing
from typing import List
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as csp


def initialize_gpu(gpu_id):
    """
    GPUの初期化
    """
    cp.cuda.Device(gpu_id).use()


def prox_l122(Y: cp.ndarray, gamma: float, N: int) -> cp.ndarray:
    """
    L1,2ノルムの2乗のprox
    """
    factor = (2 * gamma) / (1 + 2 * gamma * N)
    l1_norms = cp.sum(cp.absolute(Y), axis=1)
    X = cp.sign(Y) * cp.maximum(cp.absolute(Y) - factor * l1_norms[:, None], 0)
    return X


def fista_chunk(
    gpu_id: int,
    F: np.ndarray,
    G_chunk: np.ndarray,
    lmd: float,
    gamma: float,
    max_iter: int,
    t_memo: np.ndarray,
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
    F_gpu = cp.asarray(F)
    G_chunk_gpu = cp.asarray(G_chunk)

    for i in range(max_iter):
        H_chunk_old = H_chunk.copy()
        H_chunk = prox_l122(Y_chunk - gamma * (Y_chunk @ F_gpu - G_chunk_gpu) @ F_gpu.T, gamma * lmd, N)
        Y_chunk = H_chunk + ((t_memo[i] - 1) / t_memo[i + 1]) * (H_chunk - H_chunk_old)
        if i % 10 == 0:
            print(f"Process {gpu_id+1} | Iteration {i+1}/{max_iter}")

    return csp.csr_matrix(H_chunk)


def fista_parallel(
    F: np.ndarray,
    G: np.ndarray,
    M: int,
    lmd: float,
    max_iter: int = 150,
) -> csp.csr_matrix:
    """
    FISTAの並列バージョン
    """
    gpu_ids = list(range(cp.cuda.runtime.getDeviceCount()))
    num_gpus = len(gpu_ids)
    num_processes_per_gpu = 1
    num_processes = num_gpus * num_processes_per_gpu
    N = F.shape[0]
    L = N
    gamma = 1.0 / (L * 3)

    t_memo = np.ones(max_iter + 1, dtype=np.float32)
    for i in range(1, max_iter + 1):
        t_memo[i] = (1 + math.sqrt(1 + 4 * t_memo[i - 1] ** 2)) / 2

    chunk_size = math.ceil(M / num_processes)
    chunks: List[csp.csr_matrix] = []
    futures = []

    t_start = time.perf_counter()
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
    t_end = time.perf_counter()
    print(f"Elapsed time: {t_end - t_start:.2f}s")

    H = csp.vstack(chunks).tocsr()
    return H


if __name__ == "__main__":
    print(time.strftime("%Y/%m/%d %H:%M:%S"))

    multiprocessing.set_start_method("spawn", force=True)

    cap_dates = {128: "241114", 256: "241216"}
    n = 256
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

    H_csp = fista_parallel(F_hat, G_hat, G_hat.shape[0], LAMBDA)

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
