# %% [markdown]
# ## 最適化問題
#
# $$ \min*h \|\bm{g}-F^\top \bm{h}\|\_2^2+\lambda_1\|\bm{h}\|*{1,2}^2 + \lambda*2\|D\bm{h}\|*{1,2}$$
#

# %%
import os
import time
from typing import Callable
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import package.myUtil as myUtil

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

    di_flat = di.ravel(order="F")
    dj_flat = dj.ravel(order="F")
    dk_flat = dk.ravel(order="F")
    dl_flat = dl.ravel(order="F")

    return cp.concatenate([di_flat, dj_flat, dk_flat, dl_flat])


def compute_adjoint_differences(Du):
    # Du contains concatenated differences: di, dj, dk, dl
    total_elements = M * N
    di = Du[0:total_elements].reshape(m, m, n, n, order="F")
    dj = Du[total_elements : 2 * total_elements].reshape(m, m, n, n, order="F")
    dk = Du[2 * total_elements : 3 * total_elements].reshape(m, m, n, n, order="F")
    dl = Du[3 * total_elements :].reshape(m, m, n, n, order="F")

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

    H_adj_flat = H_adj.ravel(order="F")
    return H_adj_flat


# %%
def vector2matrixCp(vector: cp.ndarray, s: int, t: int) -> cp.ndarray:
    return vector.reshape(s, t, order="F")


def mult_mass(X: cp.ndarray, h: cp.ndarray) -> cp.ndarray:
    return (h.reshape(-1, X.shape[1], order="F") @ X.T).ravel(order="F")


def mult_Dijkl(h: cp.ndarray) -> cp.ndarray:
    return compute_differences(h.reshape(M, N, order="F"))


def mult_DijklT(y: cp.ndarray) -> cp.ndarray:
    return compute_adjoint_differences(y)


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
    return X.ravel(order="F")


def prox_tv(y: cp.ndarray, gamma: float) -> cp.ndarray:
    y_reshaped = y.reshape(-1, 4, order="F")
    y_norm = cp.linalg.norm(y_reshaped, axis=1, keepdims=True)
    scaling = cp.maximum(1 - gamma / y_norm, 0)
    return (y_reshaped * scaling).ravel(order="F")


def prox_conj(prox: Callable[[cp.ndarray, float], cp.ndarray], x: cp.ndarray, gamma: float) -> cp.ndarray:
    """Conjugate proximal operator."""
    return x - gamma * prox(x / gamma, 1 / gamma)


def primal_dual_splitting(
    X: cp.ndarray, g: cp.ndarray, lambda1: float, lambda2: float, max_iter: int = ITER
) -> tuple[cp.ndarray, dict]:

    K = X.shape[0]
    N = X.shape[1]
    M = g.shape[0] // K
    print_memory_usage("Before initializing variables")
    h = cp.zeros(M * N, dtype=cp.float16)
    h_old = cp.zeros_like(h)
    y = cp.zeros(4 * M * N, dtype=cp.float16)
    y_old = cp.zeros_like(y)
    print_memory_usage("After initializing variables")

    h[:] = 1e-2
    h_old[:] = 0
    y[:] = 1e-2
    y_old[:] = 0

    tau = 1e-5
    sigma = 1e-1
    print(f"tau={tau}, sigma={sigma}")

    for k in range(max_iter):
        h_old[:] = h[:]
        y_old[:] = y[:]

        h[:] = prox_l122(
            h_old - tau * (mult_mass(X.T, (mult_mass(X, h_old) - g)) + mult_DijklT(y_old)),
            tau * lambda1,
        )

        y[:] = prox_conj(prox_tv, y_old + sigma * mult_Dijkl(2 * h - h_old), sigma * lambda2)

        if k % 10 == 9:
            primal_residual = cp.linalg.norm(h - h_old)
            dual_residual = cp.linalg.norm(y - y_old)
            print(f"iter={k}, primal_res={primal_residual:.8e}, dual_res={dual_residual:.8e}")
            if cp.isnan(primal_residual) or cp.isnan(dual_residual):
                print("NaN detected in residuals, stopping optimization.")
                break
            # if primal_residual < 1e-3 and dual_residual < 1e-3:
            #     print("Convergence criteria met.")
            #     break
        else:
            print(f"iter={k}")

        if k % 50 == 49:
            print("2nd", calculate_2nd_term(h.reshape(M, N, order="F")))
            print("3rd", calculate_3rd_term(h))

    primal_residual = cp.linalg.norm(h - h_old)
    dual_residual = cp.linalg.norm(y - y_old)
    print(f"Final iteration {k+1}, primal_res={primal_residual:.8e}, dual_res={dual_residual:.8e}")
    print_memory_usage("After optimization")

    info = {
        "iterations": k + 1,
        "primal_residual": primal_residual,
        "dual_residual": dual_residual,
    }

    return h, info


# %%
# threshold_value = 12
# image_path = DATA_PATH + "/hadamard128_cap_240814/hadamard_1.png"
# apply_noise_reduction = True
# blur_radius = 4

# indices, shape, image_array = myUtil.find_low_pixel_indices(
#     image_path,
#     threshold_value,
#     apply_noise_reduction=apply_noise_reduction,
#     blur_radius=blur_radius
# )

# print(f"Total pixels below {threshold_value}: {len(indices)}")
# print(f"Indices of pixels below {threshold_value}:")
# print(indices)

# myUtil.create_heatmap(image_array, threshold_value)

# %%
# load images
INFO = "cap_240814"
G, _ = myUtil.images_to_matrix(f"{DATA_PATH}/{IMG_NAME}{n}_{INFO}/", ratio=RATIO, resize=True, ressize=m)
F, _ = myUtil.images_to_matrix(f"{DATA_PATH}/{IMG_NAME}{n}_input/", ratio=RATIO)
print("K=", F.shape[1])
white_img = Image.open(f"{DATA_PATH}/{IMG_NAME}{n}_{INFO}/{IMG_NAME}_1.png").convert("L")
white_img = white_img.resize((m, m))
white = np.asarray(white_img).ravel() / 255
white = white[:, np.newaxis]
H1 = np.tile(white, F.shape[1])
F_hat = 2 * F - 1
G_hat = 2 * G - H1
M = G_hat.shape[0]
# G_hat = myUtil.delete_pixels(G_hat, indices)

G_vec = G_hat.ravel(order="F")

# %%
F_hat_T_gpu = cp.asarray(F_hat.T).astype(cp.int8)
g_gpu = cp.asarray(G_vec).astype(cp.float16)

print(f"F device: {F_hat_T_gpu.device}")
print(f"g device: {g_gpu.device}")
del F, G, H1, F_hat, G_hat

# %%
h, info = primal_dual_splitting(F_hat_T_gpu, g_gpu, LAMBDA1, LAMBDA2)

# %%
H = h.reshape(M, N, order="F")
# np.save(f"{DIRECTORY}/systemMatrix/H_matrix_{SETTING}.npy", H)
# print(f"Saved {DIRECTORY}/systemMatrix/H_matrix_{SETTING}.npy")

SAMPLE_NAME = "Cameraman"
sample_image = Image.open(f"{DATA_PATH}/sample_image{n}/{SAMPLE_NAME}.png").convert("L")
sample_image = cp.asarray(sample_image).ravel() / 255

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
