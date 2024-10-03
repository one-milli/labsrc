# %% [markdown]
# ## 最適化問題
# 
# $$ \min_h \frac{1}{2}\|\bm{g}-(I\otimes F^\top) \bm{h}\|_2^2+\lambda_1\|\bm{h}\|_{1,2}^2 + \lambda_2\|D\bm{h}\|_{1,2}$$
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
ITER = 800
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
def compute_Dh(h):
    tensor = h.reshape((n, n, m, m), order='F')

    di = np.zeros_like(tensor)
    dj = np.zeros_like(tensor)
    dk = np.zeros_like(tensor)
    dl = np.zeros_like(tensor)

    di[1:, :, :, :] = tensor[1:, :, :, :] - tensor[:-1, :, :, :]
    dj[:, 1:, :, :] = tensor[:, 1:, :, :] - tensor[:, :-1, :, :]
    dk[:, :, 1:, :] = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
    dl[:, :, :, 1:] = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]

    di_flat = di.ravel(order='F')
    dj_flat = dj.ravel(order='F')
    dk_flat = dk.ravel(order='F')
    dl_flat = dl.ravel(order='F')

    Dh = np.concatenate([di_flat, dj_flat, dk_flat, dl_flat])

    return Dh


def compute_Dt_y(y):
    length = n * n * m * m

    di_flat = y[0:length]
    dj_flat = y[length:2*length]
    dk_flat = y[2*length:3*length]
    dl_flat = y[3*length:4*length]

    di = di_flat.reshape((n, n, m, m), order='F')
    dj = dj_flat.reshape((n, n, m, m), order='F')
    dk = dk_flat.reshape((n, n, m, m), order='F')
    dl = dl_flat.reshape((n, n, m, m), order='F')

    h_tensor = np.zeros((n, n, m, m))

    h_tensor[:-1, :, :, :] -= di[1:, :, :, :]
    h_tensor[1:, :, :, :] += di[1:, :, :, :]

    h_tensor[:, :-1, :, :] -= dj[:, 1:, :, :]
    h_tensor[:, 1:, :, :] += dj[:, 1:, :, :]

    h_tensor[:, :, :-1, :] -= dk[:, :, 1:, :]
    h_tensor[:, :, 1:, :] += dk[:, :, 1:, :]

    h_tensor[:, :, :, :-1] -= dl[:, :, :, 1:]
    h_tensor[:, :, :, 1:] += dl[:, :, :, 1:]

    h = h_tensor.ravel(order='F')

    return h

# %%
def calculate_1st_term(Gt, Ft, Ht):
    print("calculate_1st_term start")
    return cp.linalg.norm(Gt - Ft @ Ht) ** 2


def calculate_2nd_term(H):
    print("calculate_2nd_term start")
    column_sums = cp.sum(cp.abs(H), axis=1)
    result = cp.sum(column_sums**2)
    return result


def calculate_3rd_term(h):
    print("calculate_3rd_term start")
    Du = compute_Dh(h)
    Du = Du.reshape(-1, 4, order="F")
    tv = cp.sum(cp.linalg.norm(Du, axis=1))
    return tv

# %%
def prox_l122(Ht: cp.ndarray, gamma: float) -> cp.ndarray:
    l1_norms = cp.sum(cp.absolute(Ht), axis=1)
    factor = (2 * gamma) / (1 + 2 * gamma * N)
    X = cp.zeros_like(Ht)
    X = cp.sign(Ht) * cp.maximum(cp.absolute(Ht) - factor * l1_norms[:, None], 0)
    return X


def prox_tv(y: cp.ndarray, gamma: float) -> cp.ndarray:
    l2 = cp.linalg.norm(y.reshape(-1, 2, order="F"), axis=1, keepdims=True)
    l2 = cp.maximum(1 - gamma / (l2 + 1e-16), 0)
    return (l2 * y.reshape(-1, 2, order="F")).ravel(order="F")


def prox_conj(y: cp.ndarray, prox: Callable[[cp.ndarray, float], cp.ndarray], gamma: float) -> cp.ndarray:
    """Conjugate proximal operator."""
    return y - gamma * prox(y / gamma, 1 / gamma)


def primal_dual_splitting(
    Ft: cp.ndarray, Gt: cp.ndarray, lambda1: float, lambda2: float, max_iter: int = ITER
) -> tuple[cp.ndarray, dict]:

    N = Ft.shape[1]
    M = Gt.shape[1]
    print_memory_usage("Before initializing variables")
    Ht = cp.zeros(N, M, dtype=cp.float32)
    Ht_old = cp.zeros_like(Ht)
    y = cp.zeros(4 * N * M, dtype=cp.float32)
    y_old = cp.zeros_like(y)
    print_memory_usage("After initializing variables")

    Ht[:] = 1e-2
    Ht_old[:] = 0
    y[:] = 1e-2
    y_old[:] = 0

    tau = 1e-4
    sigma = 1e-4
    print(f"tau={tau}, sigma={sigma}")

    for k in range(max_iter):
        Ht_old[:] = Ht[:]
        y_old[:] = y[:]

        Ht[:] = prox_l122(
            Ht_old - tau * (Ft.T @ (Ft @ Ht - Gt) + (compute_Dt_y(y_old)).reshape(N, M, order="F")),
            lambda1 * tau,
        )

        y[:] = prox_conj(y_old + sigma * compute_Dh((2 * Ht - Ht_old).ravel(order="F")), prox_tv, lambda2 / sigma)

        if k % 100 == 99:
            primal_residual = cp.linalg.norm(Ht - Ht_old)
            dual_residual = cp.linalg.norm(y - y_old)
            print(f"iter={k}, primal_res={primal_residual:.8e}, dual_res={dual_residual:.8e}")
            print("1st", calculate_1st_term(Gt, Ft, Ht))
            print("2nd", calculate_2nd_term(Ht))
            print("3rd", calculate_3rd_term(Ht))
            if cp.isnan(primal_residual) or cp.isnan(dual_residual):
                print("NaN detected in residuals, stopping optimization.")
                break
            if primal_residual < 1e-3 and dual_residual < 1e-3:
                print("Convergence criteria met.")
                break
        else:
            print(f"iter={k}")

    primal_residual = cp.linalg.norm(Ht - Ht_old)
    dual_residual = cp.linalg.norm(y - y_old)
    print(f"Final iteration {k+1}, primal_res={primal_residual:.8e}, dual_res={dual_residual:.8e}")
    print_memory_usage("After optimization")

    info = {
        "iterations": k + 1,
        "primal_residual": primal_residual,
        "dual_residual": dual_residual,
    }

    return Ht, info

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
sample_image = Image.open(f"{DATA_PATH}/sample_image{n}/{SAMPLE_NAME}.png").convert('L')
sample_image = cp.asarray(sample_image).ravel() / 255

Hf = H @ sample_image
Hf_img = cp.asnumpy(Hf.reshape(m, m))
Hf_img = np.clip(Hf_img, 0, 1)
Hf_pil = Image.fromarray((Hf_img * 255).astype(np.uint8), mode='L')

FILENAME = f"{SAMPLE_NAME}_{SETTING}.png"
fig, ax = plt.subplots(figsize=Hf_img.shape[::-1], dpi=1, tight_layout=True)
ax.imshow(Hf_pil, cmap='gray')
ax.axis('off')
fig.savefig(f"{DIRECTORY}/{FILENAME}", dpi=1)
# plt.show()
print(f"Saved {DIRECTORY}/{FILENAME}")


