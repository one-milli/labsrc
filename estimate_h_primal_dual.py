# %% [markdown]
# ## 最適化問題
#
# $$ \min_h \|\bm{g}-F^\top \bm{h}\|_2^2+\lambda_1\|\bm{h}\|_{1,2}^2 + \lambda_2\|D\bm{h}\|_{1,2}$$
#

# %%
import os
import re
import random
import time
from typing import Callable
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import sparse

# %%
# パラメータ設定
n = 128
m = 256
N = n**2
M = m**2
LAMBDA1 = 10
LAMBDA2 = 1
SEED = 5
RATIO = 0.1
ITER = 500
# DATA_PATH = "../../OneDrive - m.titech.ac.jp/Lab/data"
DATA_PATH = "../data"
IMG_NAME = "hadamard"
DIRECTORY = DATA_PATH + "/240825"
SETTING = f"{IMG_NAME}_pr-du_p-{int(100*RATIO)}_lmd1-{LAMBDA1}_lmd2-{LAMBDA2}"

if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)
if not os.path.exists(DIRECTORY + "/systemMatrix"):
    os.makedirs(DIRECTORY + "/systemMatrix")

# %%
Di = sparse.eye(M, format="lil") - sparse.eye(M, k=m, format="lil")
Di[-m:, :] = 0
Di = Di.tocsr()

Dj = sparse.eye(M, format="lil") - sparse.eye(M, k=1, format="lil")
for p in range(1, m + 1):
    Dj[m * p - 1, m * p - 1] = 0
    if p < m:
        Dj[m * p - 1, m * p] = 0
Dj = Dj.tocsr()

Dk = sparse.eye(N, format="lil") - sparse.eye(N, k=n, format="lil")
Dk = sparse.lil_matrix(Dk[: n * (n - 1), :N])
Dk = sparse.vstack([Dk, sparse.lil_matrix((n, N))])
Dk = Dk.tocsr()

Dl = sparse.eye(N, format="lil") - sparse.eye(N, k=1, format="lil")
for p in range(1, n + 1):
    Dl[n * p - 1, n * p - 1] = 0
    if p < n:
        Dl[n * p - 1, n * p] = 0
Dl = Dl.tocsr()

# %%
Di_gpu = cp.sparse.csr_matrix(Di).astype(cp.float32)
Dj_gpu = cp.sparse.csr_matrix(Dj).astype(cp.float32)
Dk_gpu = cp.sparse.csr_matrix(Dk).astype(cp.float32)
Dl_gpu = cp.sparse.csr_matrix(Dl).astype(cp.float32)


# %%
def matrix2vectorNp(matrix: np.ndarray) -> np.ndarray:
    return matrix.reshape(-1, 1, order="F").flatten().astype(np.float16)


def matrix2vectorCp(matrix: cp.ndarray) -> cp.ndarray:
    return matrix.reshape(-1, 1, order="F").flatten().astype(cp.float16)


def vector2matrixNp(vector: np.ndarray, s: int, t: int) -> np.ndarray:
    return vector.reshape(s, t, order="F").astype(np.float16)


def vector2matrixCp(vector: cp.ndarray, s: int, t: int) -> cp.ndarray:
    return vector.reshape(s, t, order="F").astype(cp.float16)


def mult_mass(X: cp.ndarray, h: cp.ndarray, M: int) -> cp.ndarray:
    F_gpu = X.T.astype(cp.float16)
    H_gpu = cp.asarray(h.reshape(M, -1, order="F"))
    res_gpu = H_gpu @ F_gpu
    return matrix2vectorCp(res_gpu)


def mult_Dijkl(h: cp.ndarray, memptr) -> cp.ndarray:
    H = vector2matrixCp(h, M, N)
    res_gpu = cp.ndarray((4 * M, N), dtype=cp.float16, memptr=memptr)
    res_gpu[:] = cp.hstack([Di_gpu @ H, Dj_gpu @ H, H @ Dk_gpu.T, H @ Dl_gpu.T])
    return matrix2vectorCp(res_gpu)


def mult_DijklT(y: cp.ndarray, memptr) -> cp.ndarray:
    y1 = y[: M * N]
    y2 = y[M * N : 2 * M * N]
    y3 = y[2 * M * N : 3 * M * N]
    y4 = y[3 * M * N :]
    Y1 = vector2matrixCp(y1, M, N)
    Y2 = vector2matrixCp(y2, M, N)
    Y3 = vector2matrixCp(y3, M, N)
    Y4 = vector2matrixCp(y4, M, N)

    res_gpu = cp.ndarray((M, N), dtype=cp.float16, memptr=memptr)
    res_gpu[:] = Di_gpu.T @ Y1 + Dj_gpu.T @ Y2 + Y3 @ Dk_gpu.T + Y4 @ Dl_gpu.T
    return matrix2vectorCp(res_gpu)


def images_to_matrix(folder_path, convert_gray=True, rand=True, ratio=RATIO):
    files = os.listdir(folder_path)
    files.sort(key=lambda f: int(re.search(f"{IMG_NAME}_(\d+).png", f).group(1)))
    if rand:
        random.seed(SEED)
        random.shuffle(files)

    total_files = len(files)
    number_of_files_to_load = int(total_files * ratio)
    selected_files = files[:number_of_files_to_load]
    selected_files.sort(key=lambda f: int(re.search(f"{IMG_NAME}_(\d+).png", f).group(1)))

    images = []
    use_list = []

    for file in selected_files:
        index = int(re.sub(r"\D", "", file))
        use_list.append(index)
        img = Image.open(os.path.join(folder_path, file))
        if convert_gray:
            img = img.convert("L")
        img_array = np.asarray(img).flatten()
        img_array = img_array / 255
        images.append(img_array)

    return np.column_stack(images), use_list


# %%
def calculate_2nd_term(H):
    """
    Args:
        H: cupy.ndarray
    Returns:
        Scalar
    """
    # 各行ごとの絶対値の和を計算
    column_sums = cp.sum(cp.abs(H), axis=1)
    # 列ごとの和の2乗を計算し、その総和を求める
    result = cp.sum(column_sums**2)

    return result


def calculate_3rd_term(h):
    Du = mult_Dijkl(h)

    d_i = Du[0 : M * N - 1, :]
    d_j = Du[M * N : 2 * M * N - 1, :]
    d_k = Du[2 * M * N : 3 * M * N - 1, :]
    d_l = Du[3 * M * N :, :]

    tv = cp.sum(cp.sqrt(d_i**2 + d_j**2 + d_k**2 + d_l**2))

    return tv


# %%
def prox_l1(y: cp.ndarray, tau: float) -> cp.ndarray:
    return cp.sign(y) * cp.maximum(cp.absolute(y) - tau, 0)


def prox_l122(y: cp.ndarray, gamma: float) -> cp.ndarray:
    Y = cp.asarray(vector2matrixCp(y, M, N))
    l1_norms = cp.sum(cp.absolute(Y), axis=1)
    factor = (2 * gamma) / (1 + 2 * gamma * N)
    X = cp.zeros_like(Y)
    X = cp.sign(Y) * cp.maximum(cp.absolute(Y) - factor * l1_norms[:, None], 0)
    return matrix2vectorCp(X)


def prox_tv(y: cp.ndarray, gamma: float) -> cp.ndarray:
    Dx_norm = cp.linalg.norm(y.reshape(-1, 4, order="F"), axis=1).astype(cp.float16)
    Dx_norm = cp.tile(Dx_norm[:, None], (1, 4))

    prox = cp.maximum(1 - gamma / Dx_norm, 0) * y.reshape(-1, 4, order="F")
    prox = prox.reshape(-1, order="F")

    return prox


def prox_conj(prox: Callable[[cp.ndarray, float], cp.ndarray], x: cp.ndarray, gamma: float) -> cp.ndarray:
    """Conjugate proximal operator."""
    return x - gamma * prox(x / gamma, 1 / gamma)


def primal_dual_splitting(
    X: cp.ndarray, g: cp.ndarray, lambda1: float, lambda2: float, max_iter: int = ITER
) -> tuple[np.ndarray, dict]:
    """
    Solve the optimization problem:
    min_h ||g-Xh||_2^2 + lambda1 P(h) + lambda2 ||Dh||_{1,2}
    using the primal-dual splitting method.

    Args:
        X (cp.ndarray): Matrix X in the problem formulation.
        g (cp.ndarray): Vector g in the problem formulation.
        lambda1 (float): Regularization parameter for L1 norm of h.
        lambda2 (float): Regularization parameter for L1 norm of Dh.

    Returns:
        tuple[np.ndarray, dict]: Solution h and a dictionary containing additional information.
    """

    h = cp.ndarray((M * N,), dtype=cp.float16, memptr=cp.cuda.malloc_managed(M * N * 2))
    h_old = cp.ndarray((M * N,), dtype=cp.float16, memptr=cp.cuda.malloc_managed(M * N * 2))
    y = cp.ndarray((4 * M * N,), dtype=cp.float16, memptr=cp.cuda.malloc_managed(4 * M * N * 2))
    y_old = cp.ndarray((4 * M * N,), dtype=cp.float16, memptr=cp.cuda.malloc_managed(4 * M * N * 2))
    memptr_D = cp.cuda.malloc_managed(4 * M * N * 2)
    memptr_DT = cp.cuda.malloc_managed(M * N * 2)

    h[:] = 0
    h_old[:] = 0
    y[:] = 0
    y_old[:] = 0

    # Compute Lipschitz constant of grad_f
    tau = 1 / (4096 * 3)
    sigma = 1 / (16384 * 3)
    print(f"tau={tau}, sigma={sigma}")

    start = time.perf_counter()
    for k in range(max_iter):
        h_old = h.copy()
        y_old = y.copy()

        h = prox_l122(
            h_old - tau * (mult_mass(X.T, (mult_mass(X, h_old, M) - g), M) - mult_DijklT(y_old, memptr_DT)),
            tau * lambda1,
        )

        y = prox_conj(prox_tv, y_old + sigma * mult_Dijkl(2 * h - h_old, memptr_D), sigma / lambda2)

        # calculate 2nd term & 3rd term
        if k % 100 == 0:
            print("2nd", calculate_2nd_term(vector2matrixCp(h, M, N)))
            print("3rd", calculate_3rd_term(h))

        if k == max_iter - 1:
            primal_residual = cp.linalg.norm(h - h_old)
            dual_residual = cp.linalg.norm(y - y_old)
            print(f"iter={k}, primal_res={primal_residual:.4f}, dual_res={dual_residual:.4f}")
            break
        else:
            print(f"iter={k}")

    end = time.perf_counter()
    info = {
        "iterations": k + 1,
        "primal_residual": primal_residual,
        "dual_residual": dual_residual,
        "time": end - start,
    }

    del h, h_old, y, y_old

    return cp.asnumpy(h), info


# %%
# load images
INFO = "cap_240814"
G, use = images_to_matrix(f"{DATA_PATH}/{IMG_NAME}{n}_{INFO}/")
F, _ = images_to_matrix(f"{DATA_PATH}/{IMG_NAME}{n}_input/")
print("K=", F.shape[1])
white_img = Image.open(f"{DATA_PATH}/{IMG_NAME}{n}_{INFO}/{IMG_NAME}_1.png").convert("L")
white = np.asarray(white_img).flatten() / 255
white = white[:, np.newaxis]
H1 = np.tile(white, F.shape[1])
F_hat = 2 * F - 1
G_hat = 2 * G - H1

g = matrix2vectorNp(G_hat)

# %%
F_hat_T_gpu = cp.asarray(F_hat.T).astype(cp.int8)
g_gpu = cp.asarray(g).astype(cp.float16)
del F, G, H1, F_hat, G_hat

# %%
h, info = primal_dual_splitting(F_hat_T_gpu, g_gpu, LAMBDA1, LAMBDA2)

# %%
H = vector2matrixNp(h, M, N)
np.save(f"{DIRECTORY}/systemMatrix/H_matrix_{SETTING}.npy", H)
print(f"Saved {DIRECTORY}/systemMatrix/H_matrix_{SETTING}.npy")

SAMPLE_NAME = "Cameraman"
sample_image = Image.open(f"{DATA_PATH}/sample_image{n}/{SAMPLE_NAME}.png").convert("L")
sample_image = np.asarray(sample_image).flatten() / 255

Hf = H @ sample_image
Hf_img = Hf.reshape(m, m)
Hf_img = np.clip(Hf_img, 0, 1)
Hf_pil = Image.fromarray((Hf_img * 255).astype(np.uint8), mode="L")

FILENAME = f"{SAMPLE_NAME}_{SETTING}.png"
fig, ax = plt.subplots(figsize=Hf_img.shape[::-1], dpi=1, tight_layout=True)
ax.imshow(Hf_pil, cmap="gray")
ax.axis("off")
fig.savefig(f"{DIRECTORY}/{FILENAME}", dpi=1)
plt.show()

# %%
# H_true = np.load(f"{DATA_PATH}/systemMatrix/H_matrix_true.npy")
# rem = np.linalg.norm(H_true - H, "fro")
# print(rem)