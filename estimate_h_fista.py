# %%
import os
import time
from typing import Callable
import cupy as cp
import numpy as np
from cupyx.scipy import sparse as cps
import matplotlib.pyplot as plt
from PIL import Image
import package.myUtil as myUtil

# %%
n = 128
m = 128
N = n**2
M = m**2
LAMBDA = 1
RATIO = 0.05
# DATA_PATH = "../../OneDrive - m.titech.ac.jp/Lab/data"
DATA_PATH = "../data"
IMG_NAME = "hadamard"
DIRECTORY = DATA_PATH + "/241022"
REG = ""
SETTING = f"{IMG_NAME}_FISTA_{REG}p-{int(100*RATIO)}_lmd-{LAMBDA}_m-{m}"

if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)
if not os.path.exists(DIRECTORY + "/systemMatrix"):
    os.makedirs(DIRECTORY + "/systemMatrix")


# %%
def vector2matrixCp(vector: cp.ndarray, s: int, t: int) -> cp.ndarray:
    return vector.reshape(s, t, order="F")


def mult_mass(X: cp.ndarray, h: cp.ndarray) -> cp.ndarray:
    return (h.reshape(-1, X.shape[1], order="F") @ X.T).ravel(order="F")


# %%
def prox_l1(y: cp.ndarray, tau: float) -> cp.ndarray:
    return cp.sign(y) * cp.maximum(cp.absolute(y) - tau, 0)


def prox_l122(y: cp.ndarray, gamma: float) -> cp.ndarray:
    Y = cp.asarray(vector2matrixCp(y, M, N)).astype(cp.float32)
    l1_norms = cp.sum(cp.absolute(Y), axis=1)
    factor = (2 * gamma) / (1 + 2 * gamma * N)
    X = cp.zeros_like(Y)
    X = cp.sign(Y) * cp.maximum(cp.absolute(Y) - factor * l1_norms[:, None], 0)
    return X.flatten(order="F")


def fista(
    Ft: cp.ndarray,
    g: cp.ndarray,
    lmd: float,
    prox: Callable[[cp.ndarray, float], cp.ndarray],
    max_iter: int = 500,
    tol: float = 1e-3,
) -> cp.ndarray:
    """
    Solve the optimization problem using FISTA:
    min_h ||g - Xh||_2^2 + lambda * ||h||_1

    Parameters:
    - Ft: numpy array, the matrix Ft
    - g: numpy array, the vector g
    - lmd: float, the regularization parameter

    Returns:
    - h: numpy array, the solution vector h
    """
    K = Ft.shape[0]
    N = Ft.shape[1]
    M = g.shape[0] // K
    t = 1
    h = cp.zeros(M * N, dtype=cp.float32)
    h_old = cp.zeros_like(h)
    y = cp.zeros_like(h)
    # h = cps.csr_matrix((M * N, 1), dtype=cp.float32)
    # h_old = cps.zeros_like(h)
    # y = cps.zeros_like(h)

    # Lipschitz constant
    # L = np.linalg.norm(Ft.T @ Ft, ord=2) * 3
    gamma = 1 / (4096 * 3)

    start = time.perf_counter()
    for i in range(max_iter):
        t_old = t
        h_old = h.copy()

        h = prox(y - gamma * mult_mass(Ft.T, (mult_mass(Ft, y) - g)), gamma * lmd)
        # 加速ステップ
        t = (1 + np.sqrt(1 + 4 * t_old**2)) / 2
        y = h + (t_old - 1) / t * (h - h_old)

        error = cp.linalg.norm(h - h_old) / cp.linalg.norm(h)

        print(f"iter: {i}, error: {error}")
        if error < tol:
            break

    end = time.perf_counter()
    print(f"Elapsed time: {end-start}")

    return h


# %%
# load images
# INFO = "cap_R_230516_128"
INFO = "cap_240814"
G, _ = myUtil.images_to_matrix(f"{DATA_PATH}/{IMG_NAME}{n}_{INFO}/", ratio=RATIO, resize=True, ressize=m)
F, _ = myUtil.images_to_matrix(f"{DATA_PATH}/{IMG_NAME}{n}_input/", ratio=RATIO)
K = F.shape[1]
print("K=", K)
white_img = Image.open(f"{DATA_PATH}/{IMG_NAME}{n}_{INFO}/{IMG_NAME}_1.png").convert("L")
white_img = white_img.resize((m, m))
white = np.asarray(white_img).ravel() / 255
white = white[:, np.newaxis]
H1 = np.tile(white, F.shape[1])
F_hat = 2 * F - 1
G_hat = 2 * G - H1

g = G_hat.flatten(order="F")

# %%
F_hat_T_gpu = cp.asarray(F_hat.T).astype(cp.float32)
g_gpu = cp.asarray(g).astype(cp.float32)
del F, G, H1, F_hat, G_hat

# %%
h = fista(F_hat_T_gpu, g_gpu, LAMBDA, prox_l122)

# %%
H = h.reshape(g.shape[0] // K, N, order="F")  # cupy
np.save(f"{DIRECTORY}/systemMatrix/H_matrix_{SETTING}.npy", H)
print(f"Saved {DIRECTORY}/systemMatrix/H_matrix_{SETTING}.npy")

SAMPLE_NAME = "Cameraman"
sample_image = Image.open(f"{DATA_PATH}/sample_image{n}/{SAMPLE_NAME}.png").convert("L")
sample_image = cp.asarray(sample_image).flatten() / 255

Hf = H @ sample_image
Hf_img = cp.asnumpy(Hf.reshape(m, m))
Hf_img = np.clip(Hf_img, 0, 1)
Hf_pil = Image.fromarray((Hf_img * 255).astype(np.uint8), mode="L")

FILENAME = f"{SAMPLE_NAME}_{SETTING}.png"
fig, ax = plt.subplots(figsize=Hf_img.shape[::-1], dpi=1, tight_layout=True)
ax.imshow(Hf_pil, cmap="gray")
ax.axis("off")
fig.savefig(f"{DIRECTORY}/{FILENAME}", dpi=1)
# plt.show()
print(f"Saved {DIRECTORY}/{FILENAME}")
