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
DATA_PATH = "../../OneDrive - m.titech.ac.jp/Lab/data"
# DATA_PATH = "../data"
IMG_NAME = "hadamard"
DIRECTORY = DATA_PATH + "/241022"
SETTING = f"{IMG_NAME}_FISTA_p-{int(100*RATIO)}_lmd-{LAMBDA}_m-{m}"

if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)
if not os.path.exists(DIRECTORY + "/systemMatrix"):
    os.makedirs(DIRECTORY + "/systemMatrix")


# %%
def prox_l122(Y: cp.ndarray, gamma: float) -> cps.csr_matrix:
    l1_norms = cp.sum(cp.absolute(Y), axis=1)
    factor = (2 * gamma) / (1 + 2 * gamma * N)
    return cps.csr_matrix(cp.sign(Y) * cp.maximum(cp.absolute(Y) - factor * l1_norms[:, None], 0))


def fista(
    Ft: cp.ndarray,
    Gt: cp.ndarray,
    lmd: float,
    prox: Callable[[cp.ndarray, float], cp.ndarray],
    max_iter: int = 500,
    tol: float = 1e-3,
) -> cp.ndarray:
    """
    Solve the optimization problem using FISTA:
    min_h ||g - Xh||_2^2 + lambda * ||h||_1,2^2

    Parameters:
    - Ft: numpy array, the matrix Ft
    - g: numpy array, the vector g
    - lmd: float, the regularization parameter

    Returns:
    - h: numpy array, the solution vector h
    """
    N = Ft.shape[1]
    M = Gt.shape[1]
    t = 1
    # Ht = cp.zeros((N, M), dtype=cp.float32)
    # Ht_old = cp.zeros_like(Ht)
    # Yt = cp.zeros_like(Ht)
    Ht = cps.csr_matrix((N, M), dtype=cp.float32)
    Ht_old = cps.csr_matrix((N, M), dtype=cp.float32)
    Yt = cps.csr_matrix((N, M), dtype=cp.float32)

    # Lipschitz constant
    # L = np.linalg.norm(Ft.T @ Ft, ord=2) * 3
    gamma = 1 / (4096 * 3)
    # H_true = cp.load(f"{DIRECTORY}/systemMatrix/H_matrix_hadamard_gf.npy")
    ff = Ft.T @ Ft

    start = time.perf_counter()
    for i in range(max_iter):
        t_old = t
        Ht_old = Ht.copy()

        # A_nonsparse = Yt - gamma * Ft.T @ (Ft @ Ht - Gt)
        A_nonsparse = ff @ Ht
        A_nonsparse = Yt - gamma * A_nonsparse
        A_nonsparse = A_nonsparse - Ft.T @ Gt
        Ht = cps.csr_matrix(prox(A_nonsparse, gamma * lmd))
        t = (1 + np.sqrt(1 + 4 * t_old**2)) / 2
        Yt = Ht + ((t_old - 1) / t) * (Ht - Ht_old)

        if i % 100 == 99:
            error = cps.linalg.norm(Ht - Ht_old) / cps.linalg.norm(Ht)
            print(f"iter: {i}, error: {error}")
            # rem = cp.linalg.norm(Ht - H_true.T)
            # print(f"iter: {i}, error: {error}, rem: {rem}")
            if error < tol:
                break
        else:
            print(f"iter: {i}")

    end = time.perf_counter()
    print(f"Elapsed time: {end-start}")

    return Ht


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

# %%
F_hat_T_gpu = cp.asarray(F_hat.T).astype(cp.int8)
G_hat_T_gpu = cp.asarray(G_hat.T).astype(cp.float32)
del F, G, H1, F_hat, G_hat

# %%
Ht = fista(F_hat_T_gpu, G_hat_T_gpu, LAMBDA, prox_l122)

# %%
H = Ht.T
del Ht
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
print(f"Saved {DIRECTORY}/{FILENAME}")
