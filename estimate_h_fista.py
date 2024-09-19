# %%
import os
import re
import random
import time
from typing import Callable
import cupy as cp
import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.io as sio
from cupyx.scipy import sparse as cps
import matplotlib.pyplot as plt
from PIL import Image

# %%
n = 128
m = 256
N = n**2
M = m**2
LAMBDA = 100
SEED = 5
RATIO = 0.1
# DATA_PATH = "../../OneDrive - m.titech.ac.jp/Lab/data"
DATA_PATH = "../data"
IMG_NAME = "hadamard"
DIRECTORY = DATA_PATH + "/240825"
REG = "l1"
SETTING = f"{IMG_NAME}_FISTA_{REG}_p-{int(100*RATIO)}_lmd-{LAMBDA}"

if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)
if not os.path.exists(DIRECTORY + "/systemMatrix"):
    os.makedirs(DIRECTORY + "/systemMatrix")


# %%
def vector2matrixCp(vector: cp.ndarray, s: int, t: int) -> cp.ndarray:
    return vector.reshape(s, t, order="F").astype(cp.float32)


def mult_mass(X: cp.ndarray, h: cp.ndarray, M: int) -> cp.ndarray:
    H_gpu = cp.asarray(h.reshape(M, -1, order="F"))
    return (H_gpu @ X.T).flatten(order="F")


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
    X: cp.ndarray,
    g: cp.ndarray,
    lmd: float,
    prox: Callable[[cp.ndarray, float], cp.ndarray],
    max_iter: int = 500,
    tol: float = 1e-2,
) -> np.ndarray:
    """
    Solve the optimization problem using FISTA:
    min_h ||g - Xh||_2^2 + lambda * ||h||_1

    Parameters:
    - X: numpy array, the matrix X
    - g: numpy array, the vector g
    - lmd: float, the regularization parameter

    Returns:
    - h: numpy array, the solution vector h
    """
    t = 1
    h = cp.ones(X.shape[1], dtype=cp.float32)
    h_old = cp.ones_like(h)
    y = cp.ones_like(h)
    y_old = cp.ones_like(y)

    # Lipschitz constant
    # L = np.linalg.norm(X.T @ X, ord=2) * 3
    gamma = 1 / (4096 * 3)
    switched_to_sparse = False
    sparsity_threshold = 1e-6
    switch_to_sparse_iter = 100

    start = time.perf_counter()
    for i in range(max_iter):
        t_old = t
        h_old = h.copy()
        y_old = y.copy()

        t = (1 + np.sqrt(1 + 4 * t_old**2)) / 2
        h = prox(y_old - gamma * mult_mass(X.T, (mult_mass(X, y_old, M) - g), M), gamma * lmd)
        h = cp.where(cp.abs(h) < sparsity_threshold, 0, h)
        if not switched_to_sparse and i >= switch_to_sparse_iter:
            sparsity_level = cp.count_nonzero(h) / h.size
            if sparsity_level < 0.1:  # 非ゼロ要素が全体の10%未満の場合
                h = cps.csr_matrix(h)
                y = cps.csr_matrix(y)
                grad = cps.csr_matrix(grad)
                switched_to_sparse = True
                print(f"Iteration {i}: Switched to sparse representation.")
        y = h + (t_old - 1) / t * (h - h_old)

        if switched_to_sparse:
            error = cp.linalg.norm((y - y_old).toarray()) / cp.linalg.norm(y.toarray())
        else:
            error = cp.linalg.norm(y - y_old) / cp.linalg.norm(y)

        print(f"iter: {i}, error: {error}")
        if error < tol:
            break

    end = time.perf_counter()
    print(f"Elapsed time: {end-start}")
    if switched_to_sparse:
        h = h.toarray().flatten()

    return cp.asnumpy(y)


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

g = G_hat.flatten(order="F")

# %%
F_hat_T_gpu = cp.asarray(F_hat.T).astype(cp.float32)
g_gpu = cp.asarray(g).astype(cp.float32)
del F, G, H1, F_hat, G_hat

# %%
h = fista(F_hat_T_gpu, g_gpu, LAMBDA, prox_l1)

# %%
H = h.reshape(G_hat.shape[0], N, order="F")
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
