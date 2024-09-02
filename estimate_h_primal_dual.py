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
import cupyx.scipy.sparse as csp

# %%
n = 128
m = 192
N = n**2
M = m**2
LAMBDA1 = 1e1
LAMBDA2 = 1e1
SEED = 5
RATIO = 0.05
ITER = 300
DATA_PATH = "../data"
IMG_NAME = "hadamard"
DIRECTORY = DATA_PATH + "/240825"
SETTING = f"{IMG_NAME}_pr-du_p-{int(100*RATIO)}_lmd1-{LAMBDA1}_lmd2-{LAMBDA2}"

if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)
if not os.path.exists(DIRECTORY + "/systemMatrix"):
    os.makedirs(DIRECTORY + "/systemMatrix")

cp.cuda.Device(1).use()
cp.cuda.Device(2).use()
cp.cuda.Device(3).use()

# %%
Di_gpu = csp.eye(M, dtype=cp.float32, format='csr') - \
    csp.eye(M, k=m, dtype=cp.float32, format='csr')
Di_gpu[-m:, :] = 0

Dj_gpu = csp.eye(M, dtype=cp.float32, format='csr') - \
    csp.eye(M, k=1, dtype=cp.float32, format='csr')
for p in range(1, m + 1):
    Dj_gpu[m * p - 1, m * p - 1] = 0
    if p < m:
        Dj_gpu[m * p - 1, m * p] = 0

Dk_gpu = csp.eye(N, dtype=cp.float32, format='csr') - \
    csp.eye(N, k=n, dtype=cp.float32, format='csr')
Dk_gpu = csp.csr_matrix(Dk_gpu[:n * (n - 1), :N])
Dk_gpu = csp.vstack([Dk_gpu, csp.csr_matrix((n, N))])

Dl_gpu = csp.eye(N, dtype=cp.float32, format='csr') - \
    csp.eye(N, k=1, dtype=cp.float32, format='csr')
for p in range(1, n + 1):
    Dl_gpu[n * p - 1, n * p - 1] = 0
    if p < n:
        Dl_gpu[n * p - 1, n * p] = 0

# %%


def matrix2vectorNp(matrix: np.ndarray) -> np.ndarray:
    return matrix.reshape(-1, 1, order="F").flatten()


def matrix2vectorCp(matrix: cp.ndarray) -> cp.ndarray:
    return matrix.reshape(-1, 1, order="F").flatten()


def vector2matrixNp(vector: np.ndarray, s: int, t: int) -> np.ndarray:
    return vector.reshape(s, t, order="F")


def vector2matrixCp(vector: cp.ndarray, s: int, t: int) -> cp.ndarray:
    return vector.reshape(s, t, order="F")


def mult_mass(X: cp.ndarray, h: cp.ndarray) -> cp.ndarray:
    return matrix2vectorCp((h.reshape(M, -1, order="F") @ X.T).astype(cp.float16))


def mult_Dijkl(h: cp.ndarray, memptr) -> cp.ndarray:
    with cp.cuda.Device(h.device.id):
        H = vector2matrixCp(h, M, N)
        res_gpu = cp.ndarray((4 * M * N), dtype=cp.float16, memptr=memptr)
        res_gpu[: M * N] = matrix2vectorCp(Di_gpu @ H)
        res_gpu[M * N: 2 * M * N] = matrix2vectorCp(Dj_gpu @ H)
        res_gpu[2 * M * N: 3 * M * N] = matrix2vectorCp(H @ Dk_gpu.T)
        res_gpu[3 * M * N:] = matrix2vectorCp(H @ Dl_gpu.T)
        return res_gpu


def mult_DijklT(y: cp.ndarray, memptr) -> cp.ndarray:
    with cp.cuda.Device(y.device.id):
        res_gpu = cp.ndarray((M, N), dtype=cp.float16, memptr=memptr)
        res_gpu[:] = Di_gpu.T @ vector2matrixCp(y[: M * N], M, N)
        res_gpu[:] += Dj_gpu.T @ vector2matrixCp(y[M * N: 2 * M * N], M, N)
        res_gpu[:] += vector2matrixCp(y[2 * M * N: 3 * M * N], M, N) @ Dk_gpu.T
        res_gpu[:] += vector2matrixCp(y[3 * M * N:], M, N) @ Dl_gpu.T
        return matrix2vectorCp(res_gpu)


def images_to_matrix(folder_path, convert_gray=True, rand=True, ratio=RATIO, resize=False):
    files = os.listdir(folder_path)
    files.sort(key=lambda f: int(
        re.search(f"{IMG_NAME}_(\d+).png", f).group(1)))
    if rand:
        random.seed(SEED)
        random.shuffle(files)

    total_files = len(files)
    number_of_files_to_load = int(total_files * ratio)
    selected_files = files[:number_of_files_to_load]
    selected_files.sort(key=lambda f: int(
        re.search(f"{IMG_NAME}_(\d+).png", f).group(1)))

    images = []
    use_list = []

    for file in selected_files:
        index = int(re.sub(r"\D", "", file))
        use_list.append(index)
        img = Image.open(os.path.join(folder_path, file))
        if convert_gray:
            img = img.convert("L")
        if resize:
            img = img.resize((m, m))
        img_array = np.asarray(img).flatten()
        img_array = img_array / 255
        images.append(img_array)

    return np.column_stack(images), use_list

# %%


def calculate_1st_term(g, X, h):
    print("calculate_1st_term start")
    return cp.linalg.norm(g - mult_mass(X, h)) ** 2


def calculate_2nd_term(H):
    print("calculate_2nd_term start")
    column_sums = cp.sum(cp.abs(H), axis=1)
    result = cp.sum(column_sums**2)
    return result


def calculate_3rd_term(h, memptr):
    print("calculate_3rd_term start")
    Du = mult_Dijkl(h, memptr)
    tv = cp.sum(
        cp.sqrt(
            (Du[0: M * N]) ** 2
            + (Du[M * N: 2 * M * N]) ** 2
            + (Du[2 * M * N: 3 * M * N]) ** 2
            + (Du[3 * M * N:]) ** 2
        )
    )
    print("calculate_3rd_term end")
    return tv

# %%


def prox_l1(y: cp.ndarray, tau: float) -> cp.ndarray:
    return cp.sign(y) * cp.maximum(cp.absolute(y) - tau, 0)


def prox_l122(y: cp.ndarray, gamma: float) -> cp.ndarray:
    l1_norms = cp.sum(cp.absolute(vector2matrixCp(y, M, N)), axis=1)
    factor = (2 * gamma) / (1 + 2 * gamma * N)
    X = cp.sign(vector2matrixCp(y, M, N)) * cp.maximum(
        cp.absolute(vector2matrixCp(y, M, N)) - factor * l1_norms[:, None], 0
    )
    return matrix2vectorCp(X)


def prox_tv(y: cp.ndarray, gamma: float) -> cp.ndarray:
    Dx_norm = cp.linalg.norm(
        y.reshape(-1, 4, order="F"), axis=1).astype(cp.float16)
    Dx_norm = cp.tile(Dx_norm[:, None], (1, 4))
    return (cp.maximum(1 - gamma / Dx_norm, 0) * y.reshape(-1, 4, order="F")).reshape(-1, order="F")


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

    with cp.cuda.Device(X.device.id):
        h = cp.ndarray((M * N,), dtype=cp.float16,
                       memptr=cp.cuda.malloc_managed(M * N * 2))
        h_old = cp.ndarray((M * N,), dtype=cp.float16,
                           memptr=cp.cuda.malloc_managed(M * N * 2))
        print(f"h GPU memory usage: {h.nbytes / 1024**2} MB")
        print(f"h_old GPU memory usage: {h_old.nbytes / 1024**2} MB")

    with cp.cuda.Device(g.device.id):
        y = cp.ndarray((4 * M * N,), dtype=cp.float16,
                       memptr=cp.cuda.malloc_managed(4 * M * N * 2))
        y_old = cp.ndarray((4 * M * N,), dtype=cp.float16,
                           memptr=cp.cuda.malloc_managed(4 * M * N * 2))
        print(f"y GPU memory usage: {y.nbytes / 1024**2} MB")
        print(f"y_old GPU memory usage: {y_old.nbytes / 1024**2} MB")

    memptr_D = cp.cuda.malloc_managed(4 * M * N * 2)
    memptr_DT = cp.cuda.malloc_managed(M * N * 2)

    h[:] = 0
    h_old[:] = 0
    y[:] = 0
    y_old[:] = 0

    # Compute Lipschitz constant of grad_f
    tau = 1 / (N * 1)
    sigma = 1 / (M * 1)
    print(f"tau={tau}, sigma={sigma}")

    # start = time.perf_counter()
    for k in range(max_iter):
        # GPU memory usage check
        h_old[:] = h[:]
        y_old[:] = y[:]

        h[:] = prox_l122(
            h_old - tau * (mult_mass(X.T, (mult_mass(X, h_old) - g)
                                     ) - mult_DijklT(y_old, memptr_DT)),
            tau * lambda1,
        )

        y[:] = prox_conj(prox_tv, y_old + sigma *
                         mult_Dijkl(2 * h - h_old, memptr_D), sigma / lambda2)

        # calculate 2nd term & 3rd term
        if k % 20 == 2:
            print("1st", calculate_1st_term(g, X, h))
            print("2nd", calculate_2nd_term(vector2matrixCp(h, M, N)))
            print("3rd", calculate_3rd_term(h, memptr_D))
            primal_residual = cp.linalg.norm(h - h_old)
            dual_residual = cp.linalg.norm(y - y_old)
            print(
                f"iter={k}, primal_res={primal_residual:.8f}, dual_res={dual_residual:.8f}")

        if k == max_iter - 1:
            primal_residual = cp.linalg.norm(h - h_old)
            dual_residual = cp.linalg.norm(y - y_old)
            print(
                f"iter={k}, primal_res={primal_residual:.8f}, dual_res={dual_residual:.8f}")
            break
        else:
            print(f"iter={k}")

    # end = time.perf_counter()
    info = {
        "iterations": k + 1,
        "primal_residual": primal_residual,
        "dual_residual": dual_residual,
        # "time": end - start,
    }

    return cp.asnumpy(h), info


# %%
# load images
INFO = "cap_240814"
G, use = images_to_matrix(f"{DATA_PATH}/{IMG_NAME}{n}_{INFO}/", resize=True)
F, _ = images_to_matrix(f"{DATA_PATH}/{IMG_NAME}{n}_input/")
print("K=", F.shape[1])
white_img = Image.open(
    f"{DATA_PATH}/{IMG_NAME}{n}_{INFO}/{IMG_NAME}_1.png").convert("L")
white_img = white_img.resize((m, m))
white = np.asarray(white_img).flatten() / 255
white = white[:, np.newaxis]
H1 = np.tile(white, F.shape[1])
F_hat = 2 * F - 1
G_hat = 2 * G - H1

g = matrix2vectorNp(G_hat)

# %%
F_hat_T_gpu = cp.asarray(F_hat.T).astype(cp.int8)
g_gpu = cp.asarray(g).astype(cp.float16)

print(f"F device: {F_hat_T_gpu.device}")
print(f"g device: {g_gpu.device}")
del F, G, H1, F_hat, G_hat

# %%
h, info = primal_dual_splitting(F_hat_T_gpu, g_gpu, LAMBDA1, LAMBDA2)

# %%
H = vector2matrixNp(h, M, N)
np.save(f"{DIRECTORY}/systemMatrix/H_matrix_{SETTING}.npy", H)
print(f"Saved {DIRECTORY}/systemMatrix/H_matrix_{SETTING}.npy")

SAMPLE_NAME = "Cameraman"
sample_image = Image.open(
    f"{DATA_PATH}/sample_image{n}/{SAMPLE_NAME}.png").convert('L')
sample_image = np.asarray(sample_image).flatten() / 255

Hf = H @ sample_image
Hf_img = Hf.reshape(m, m)
Hf_img = np.clip(Hf_img, 0, 1)
Hf_pil = Image.fromarray((Hf_img * 255).astype(np.uint8), mode='L')

FILENAME = f"{SAMPLE_NAME}_{SETTING}.png"
fig, ax = plt.subplots(figsize=Hf_img.shape[::-1], dpi=1, tight_layout=True)
ax.imshow(Hf_pil, cmap='gray')
ax.axis('off')
fig.savefig(f"{DIRECTORY}/{FILENAME}", dpi=1)
plt.show()
