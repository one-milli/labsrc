# %% [markdown]
# ## 最適化問題
# 
# $$ \min*h \|\bm{g}-F^\top \bm{h}\|\_2^2+\lambda_1\|\bm{h}\|*{1,2}^2 + \lambda*2\|D\bm{h}\|*{1,2}$$
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

# %%
n = 128
m = 192
N = n**2
M = m**2
LAMBDA1 = 1e2
LAMBDA2 = 1e2
SEED = 5
RATIO = 0.05
ITER = 500
DATA_PATH = "../data"
IMG_NAME = "hadamard"
DIRECTORY = DATA_PATH + "/240825"
SETTING = f"{IMG_NAME}_pr-du_p-{int(100*RATIO)}_lmd1-{LAMBDA1}_lmd2-{LAMBDA2}"

if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)
if not os.path.exists(DIRECTORY + "/systemMatrix"):
    os.makedirs(DIRECTORY + "/systemMatrix")

# %%
def print_memory_usage(message):
    mempool = cp.get_default_memory_pool()
    used_bytes = mempool.used_bytes()
    total_bytes = mempool.total_bytes()
    print(f"{message}: Used memory: {used_bytes / 1024**3:.2f} GB, Total memory: {total_bytes / 1024**3:.2f} GB")

# %%
def compute_differences(H):
    # H is expected to be of shape (M, N)
    # Reshape H into 4D tensor with shape (m, m, n, n)
    H = H.reshape(m, m, n, n, order="F")

    di = cp.zeros_like(H)
    dj = cp.zeros_like(H)
    dk = cp.zeros_like(H)
    dl = cp.zeros_like(H)

    di[:-1, :, :, :] = H[1:, :, :, :] - H[:-1, :, :, :]
    dj[:, :-1, :, :] = H[:, 1:, :, :] - H[:, :-1, :, :]
    dk[:, :, :-1, :] = H[:, :, 1:, :] - H[:, :, :-1, :]
    dl[:, :, :, :-1] = H[:, :, :, 1:] - H[:, :, :, :-1]

    di_flat = di.ravel(order='F')
    dj_flat = dj.ravel(order='F')
    dk_flat = dk.ravel(order='F')
    dl_flat = dl.ravel(order='F')

    return cp.concatenate([di_flat, dj_flat, dk_flat, dl_flat])


def compute_adjoint_differences(Du):
    # Du contains concatenated differences: di, dj, dk, dl
    total_elements = M * N
    di = Du[0:total_elements].reshape(m, m, n, n, order='F')
    dj = Du[total_elements:2*total_elements].reshape(m, m, n, n, order='F')
    dk = Du[2*total_elements:3*total_elements].reshape(m, m, n, n, order='F')
    dl = Du[3*total_elements:].reshape(m, m, n, n, order='F')

    H_adj = cp.zeros((m, m, n, n), dtype=Du.dtype)

    # Compute adjoint differences for di (axis 0)
    H_adj[1:, :, :, :] -= di[:-1, :, :, :]
    H_adj[:-1, :, :, :] += di[:-1, :, :, :]

    # Compute adjoint differences for dj (axis 1)
    H_adj[:, 1:, :, :] -= dj[:, :-1, :, :]
    H_adj[:, :-1, :, :] += dj[:, :-1, :, :]

    # Compute adjoint differences for dk (axis 2)
    H_adj[:, :, 1:, :] -= dk[:, :, :-1, :]
    H_adj[:, :, :-1, :] += dk[:, :, :-1, :]

    # Compute adjoint differences for dl (axis 3)
    H_adj[:, :, :, 1:] -= dl[:, :, :, :-1]
    H_adj[:, :, :, :-1] += dl[:, :, :, :-1]

    H_adj_flat = H_adj.ravel(order='F')
    return H_adj_flat

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
    return (h.reshape(M, -1, order="F") @ X.T).ravel(order="F")


def mult_Dijkl(h: cp.ndarray) -> cp.ndarray:
    return compute_differences(h.reshape(M, N, order="F"))


def mult_DijklT(y: cp.ndarray) -> cp.ndarray:
    return compute_adjoint_differences(y)


def images_to_matrix(folder_path, convert_gray=True, rand=True, ratio=RATIO, resize=False):
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

def calculate_3rd_term(h):
    print("calculate_3rd_term start")
    Du = mult_Dijkl(h)
    Du = Du.reshape(-1, 4, order="F")
    tv = cp.sum(cp.linalg.norm(Du, axis=1))
    print("calculate_3rd_term end")
    return tv

# %%
def prox_l1(y: cp.ndarray, tau: float) -> cp.ndarray:
    return cp.sign(y) * cp.maximum(cp.absolute(y) - tau, 0)


def prox_tv(y: cp.ndarray, gamma: float) -> cp.ndarray:
    y_reshaped = y.reshape(-1, 4, order="F")
    y_norm = cp.linalg.norm(y_reshaped, axis=1, keepdims=True) + 1e-15  # Avoid division by zero
    scaling = cp.maximum(1 - gamma / y_norm, 0)
    return (y_reshaped * scaling).reshape(-1, order="F")


def prox_conj(prox: Callable[[cp.ndarray, float], cp.ndarray], x: cp.ndarray, gamma: float) -> cp.ndarray:
    """Conjugate proximal operator."""
    return x - gamma * prox(x / gamma, 1 / gamma)


def primal_dual_splitting(
    X: cp.ndarray, g: cp.ndarray, lambda1: float, lambda2: float, max_iter: int = ITER
) -> tuple[np.ndarray, dict]:

    print_memory_usage("Before initializing variables")
    h = cp.zeros((M * N,), dtype=cp.float16)
    h_old = cp.zeros_like(h)
    y = cp.zeros((4 * M * N,), dtype=cp.float16)
    y_old = cp.zeros_like(y)
    print_memory_usage("After initializing variables")

    h[:] = 1e-2
    h_old[:] = 0
    y[:] = 1e-2
    y_old[:] = 0

    tau = 1e-4
    sigma = 1e-1
    print(f"tau={tau}, sigma={sigma}")

    for k in range(max_iter):
        h_old[:] = h[:]
        y_old[:] = y[:]

        h[:] = prox_l1(
            h_old - tau * (mult_mass(X.T, (mult_mass(X, h_old) - g)) - mult_DijklT(y_old)),
            tau * lambda1,
        )

        y[:] = prox_conj(prox_tv, y_old + sigma * mult_Dijkl(2 * h - h_old), sigma * lambda2)

        if k % 10 == 9:
            primal_residual = cp.linalg.norm(h - h_old) / cp.linalg.norm(h)
            dual_residual = cp.linalg.norm(y - y_old) / cp.linalg.norm(y)
            print(f"iter={k}, primal_res={primal_residual:.8e}, dual_res={dual_residual:.8e}")
            if cp.isnan(primal_residual) or cp.isnan(dual_residual):
                print("NaN detected in residuals, stopping optimization.")
                break
            if primal_residual < 1e-3 and dual_residual < 1e-3:
                print("Convergence criteria met.")
                break

        if k % 50 == 49:
            print("2nd", calculate_2nd_term(vector2matrixCp(h, M, N)))
            print("3rd", calculate_3rd_term(h))

    primal_residual = cp.linalg.norm(h - h_old) / cp.linalg.norm(h)
    dual_residual = cp.linalg.norm(y - y_old) / cp.linalg.norm(y)
    print(f"Final iteration {k+1}, primal_res={primal_residual:.8e}, dual_res={dual_residual:.8e}")
    print_memory_usage("After optimization")

    info = {
        "iterations": k + 1,
        "primal_residual": primal_residual,
        "dual_residual": dual_residual,
    }

    return cp.asnumpy(h), info

# %%
# load images
INFO = "cap_240814"
G, use = images_to_matrix(f"{DATA_PATH}/{IMG_NAME}{n}_{INFO}/", resize=True)
F, _ = images_to_matrix(f"{DATA_PATH}/{IMG_NAME}{n}_input/")
print("K=", F.shape[1])
white_img = Image.open(f"{DATA_PATH}/{IMG_NAME}{n}_{INFO}/{IMG_NAME}_1.png").convert("L")
white_img = white_img.resize((m, m))
white = np.asarray(white_img).flatten() / 255
white = white[:, np.newaxis]
H1 = np.tile(white, F.shape[1])
F_hat = 2 * F - 1
G_hat = 2 * G - H1

G_vec = matrix2vectorNp(G_hat)

# %%
F_hat_T_gpu = cp.asarray(F_hat.T).astype(cp.int8)
g_gpu = cp.asarray(G_vec).astype(cp.float16)

print(f"F device: {F_hat_T_gpu.device}")
print(f"g device: {g_gpu.device}")
del F, G, H1, F_hat, G_hat

# %%
h, info = primal_dual_splitting(F_hat_T_gpu, g_gpu, LAMBDA1, LAMBDA2)

# %%
H = vector2matrixNp(h, M, N)
# np.save(f"{DIRECTORY}/systemMatrix/H_matrix_{SETTING}.npy", H)
# print(f"Saved {DIRECTORY}/systemMatrix/H_matrix_{SETTING}.npy")

SAMPLE_NAME = "Cameraman"
sample_image = Image.open(f"{DATA_PATH}/sample_image{n}/{SAMPLE_NAME}.png").convert('L')
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
# plt.show()
print(f"Saved {DIRECTORY}/{FILENAME}")


