# %%
import os
import math
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as csp
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display
import package.myUtil as myUtil

# %%
cap_dates = {128: "241114", 256: "241205"}
n = 128
LAMBDA = 100
RATIO = 0.05
DO_THIN_OUT = True
SAVE_AS_SPARSE = True
DATA_PATH = "../data"
IMG_NAME = "hadamard"
CAP_DATE = cap_dates[n]
EXP_DATE = "241206"
DIRECTORY = f"{DATA_PATH}/{EXP_DATE}"
SETTING = f"p-{int(100*RATIO)}_lmd-{LAMBDA}"
if DO_THIN_OUT:
    SETTING = SETTING + "to"

if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)
if not os.path.exists(DIRECTORY + "/systemMatrix"):
    os.makedirs(DIRECTORY + "/systemMatrix")
use_list = myUtil.get_use_list(n * n, RATIO)

# %%
G = myUtil.images2matrix(f"{DATA_PATH}/{IMG_NAME}{n}_cap_{CAP_DATE}/", use_list, thin_out=DO_THIN_OUT)
F = myUtil.images2matrix(f"{DATA_PATH}/{IMG_NAME}{n}_input/", use_list).astype(cp.int8)
M, K = G.shape
N, K = F.shape
print("G shape:", G.shape, "F shape:", F.shape, "M=", M, "N=", N, "K=", K)
print("G max:", G.max(), "G min:", G.min(), "F max:", F.max(), "F min:", F.min())

black = myUtil.calculate_bias(M, DATA_PATH, CAP_DATE)
B = cp.tile(black[:, None], K)

G = G - B

white_img = Image.open(f"{DATA_PATH}/capture_{CAP_DATE}/White.png").convert("L")
white = (cp.asarray(white_img) / 255).astype(cp.float32)
if DO_THIN_OUT:
    white = white[::2, ::2].ravel() - black
else:
    white = white.ravel() - black
H1 = cp.tile(white[:, None], K)

F_hat = 2 * F - 1
G_hat = 2 * G - H1
g = G_hat.flatten(order="F").astype(cp.float32)
del F, G, H1, G_hat
cp._default_memory_pool.free_all_blocks()


# %%
def mult_mass(X: cp.ndarray, h: cp.ndarray) -> cp.ndarray:
    return (h.reshape(-1, X.shape[1], order="F") @ X.T).ravel(order="F")


def prox_l122(y: cp.ndarray, gamma: float, N: int) -> cp.ndarray:
    Y = cp.asarray(y.reshape(M, N, order="F")).astype(cp.float32)
    l1_norms = cp.sum(cp.absolute(Y), axis=1)
    factor = (2 * gamma) / (1 + 2 * gamma * N)
    X = cp.zeros_like(Y)
    X = cp.sign(Y) * cp.maximum(cp.absolute(Y) - factor * l1_norms[:, None], 0)
    return X.flatten(order="F")


def fista(
    Ft: cp.ndarray,
    g: cp.ndarray,
    lmd: float,
    N: int,
    M: int,
    max_iter: int = 500,
    tol: float = 1e-3,
) -> cp.ndarray:
    """
    Solve the optimization problem using FISTA:
    min_h ||g - Xh||_2^2 + lambda * ||h||_{1,2}^2
    """
    t = 1
    h = cp.zeros(M * N, dtype=cp.float32)
    h_old = cp.zeros_like(h)
    y = cp.zeros_like(h)

    # Lipschitz constant
    eigs = cp.linalg.eigvalsh(Ft.T @ Ft)
    L = cp.max(eigs)
    print("L:", L)
    gamma = 1.0 / (L * 3)

    for i in range(max_iter):
        t_old = t
        h_old = h.copy()

        a = y - gamma * mult_mass(Ft.T, (mult_mass(Ft, y) - g))
        h = prox_l122(a, gamma * lmd, N)
        print(cp.linalg.norm(h))
        t = (1 + np.sqrt(1 + 4 * t_old**2)) / 2
        y = h + ((t_old - 1) / t) * (h - h_old)

        top = cp.linalg.norm(h - h_old)
        bottom = cp.linalg.norm(h)
        error = top / bottom
        print(f"iter: {i}, error: {error} = {top} / {bottom}")
        if error < tol:
            break

    return h


# %%
h = fista(F_hat.T, g, LAMBDA, N, M)
H = h.reshape(M, N, order="F")  # cupy
print("H shape:", H.shape)
del h

# %%
if SAVE_AS_SPARSE:
    H = csp.csr_matrix(H)
    print(f"shape: {H.shape}, nnz: {H.nnz}({H.nnz / H.shape[0] / H.shape[1] * 100:.2f}%)")
    H_np = {
        "data": cp.asnumpy(H.data),
        "indices": cp.asnumpy(H.indices),
        "indptr": cp.asnumpy(H.indptr),
        "shape": H.shape,
    }
    np.savez(f"{DIRECTORY}/systemMatrix/H_matrix_{SETTING}.npz", **H_np)
    print(f"Saved {DIRECTORY}/systemMatrix/H_matrix_{SETTING}.npz")
    # myUtil.plot_sparse_matrix_cupy(H, row_range=(5500, 6000), col_range=(4500, 5000), markersize=1)
else:
    cp.save(f"{DIRECTORY}/systemMatrix/H_matrix_{SETTING}.npy", H)
    print(f"Saved {DIRECTORY}/systemMatrix/H_matrix_{SETTING}.npy")

# %%
SAMPLE_NAME = "Cameraman"
sample_image = Image.open(f"{DATA_PATH}/sample_image{n}/{SAMPLE_NAME}.png").convert("L")
sample_image = cp.asarray(sample_image).flatten() / 255

m = int(math.sqrt(M))
FILENAME = f"{SAMPLE_NAME}_{SETTING}.png"
Hf = H @ sample_image + black
Hf = cp.asnumpy(Hf.reshape(m, m))
print("Hf shape:", Hf.shape)

Hf_pil = Image.fromarray((Hf * 255).astype(np.uint8), mode="L")
Hf_pil.save(f"{DIRECTORY}/{FILENAME}", format="PNG")
print(f"Saved {DIRECTORY}/{FILENAME}")
display(Hf_pil)

plt.imshow(Hf, cmap="gray", interpolation="nearest")
plt.colorbar()
plt.title("Grayscale Heatmap")
plt.show()
