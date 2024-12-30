import os
import math
from typing import List
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as csp
from concurrent.futures import ProcessPoolExecutor

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


def prox_l122(Y: cp.ndarray, gamma: float, N: int) -> csp.csr_matrix:
    factor = (2 * gamma) / (1 + 2 * gamma * N)
    l1_norms = cp.sum(cp.absolute(Y), axis=1)
    X = cp.sign(Y) * cp.maximum(cp.absolute(Y) - factor * l1_norms[:, None], 0)
    return X


def fista_chunk(
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
    N = F.shape[0]
    rows = G_chunk.shape[0]
    H_chunk = cp.zeros((rows, N), dtype=cp.float32)
    H_chunk_old = cp.zeros((rows, N), dtype=cp.float32)
    Y_chunk = cp.zeros((rows, N), dtype=cp.float32)

    for i in range(max_iter):
        H_chunk_old = H_chunk.copy()
        H_chunk = prox_l122(Y_chunk - gamma * (Y_chunk @ F - G_chunk) @ F.T, gamma * lmd, N)
        Y_chunk = H_chunk + ((t_memo[i] - 1) / t_memo[i + 1]) * (H_chunk - H_chunk_old)

    return csp.csr_matrix(H_chunk)


def fista_parallel(
    F: cp.ndarray,
    G: cp.ndarray,
    M: int,
    lmd: float,
    chunk_size: int = 3000,
    max_iter: int = 150,
) -> csp.csr_matrix:
    """
    FISTAの並列バージョン
    """
    N = F.shape[0]
    L = N
    gamma = 1.0 / (L * 3)

    # 反復のt_memoを準備
    t_memo = cp.ones(max_iter + 1, dtype=cp.float32)
    for i in range(1, max_iter + 1):
        t_memo[i] = (1 + math.sqrt(1 + 4 * t_memo[i - 1] ** 2)) / 2

    # チャンクごとの処理を並列化
    chunks: List[csp.csr_matrix] = []
    c_length = math.ceil(M / chunk_size)
    futures = []

    with ProcessPoolExecutor() as executor:
        for c in range(0, c_length):
            print(f"Chunk {c+1}/{c_length}")
            start = c * chunk_size
            end = min((c + 1) * chunk_size, M)
            G_chunk = G[start:end, :]

            # 非同期で実行
            futures.append(executor.submit(fista_chunk, F, G_chunk, lmd, gamma, max_iter, t_memo))

        # 結果を収集
        for future in futures:
            chunks.append(future.result())

    H_csp = csp.vstack(chunks).tocsr()
    return H_csp


F_hat = cp.load(f"{DATA_PATH}/capture_{CAP_DATE}/F_hat.npy")
G_hat = cp.load(f"{DATA_PATH}/capture_{CAP_DATE}/G_hat.npy")

H = fista_parallel(F_hat, G_hat, G_hat.shape[0], LAMBDA)

print(f"shape: {H.shape}, nnz: {H.nnz}({H.nnz / H.shape[0] / H.shape[1] * 100:.2f}%)")
H_np = {"data": cp.asnumpy(H.data), "indices": cp.asnumpy(H.indices), "indptr": cp.asnumpy(H.indptr), "shape": H.shape}
np.savez(f"{DIRECTORY}/systemMatrix/H_matrix_{SETTING}.npz", **H_np)
print(f"Saved {DIRECTORY}/systemMatrix/H_matrix_{SETTING}.npz")
