# %%
import os
import numpy as np
import cupy as cp
from scipy.io import mmread
import scipy.sparse as sp
import cupyx.scipy.sparse as csp
import matplotlib.pyplot as plt
import package.myUtil as myUtil
from PIL import Image
from IPython.display import display


# %%
def gradient_operator(n):
    # 1Dの差分行列を作成
    e = cp.ones(n)
    D1d = csp.diags([-e, e], [0, 1], shape=(n - 1, n))

    # 単位行列
    I_n = csp.eye(n)

    # Kronecker積を用いて2Dの勾配演算子を作成
    Dx = csp.kron(I_n, D1d)  # 水平方向の差分
    Dy = csp.kron(D1d, I_n)  # 垂直方向の差分
    D = csp.vstack([Dx, Dy])

    return D


def divergence_operator(Dx, Dy):
    # 勾配演算子の随伴（負の発散演算子）
    DxT = -Dx.transpose()
    DyT = -Dy.transpose()
    return DxT.tocsr(), DyT.tocsr()


def proj_linf(y, tau):
    # 無限ノルムボールへの射影
    return cp.clip(y, -tau, tau)


def proj_unit_interval(f):
    # [0,1]への射影
    return cp.clip(f, 0, 1)


def prox_l1(f, gamma):
    return cp.maximum(0, cp.abs(f) - gamma) * cp.sign(f)


def prox_iota(f, gamma, min=0, max=1):
    # 非負制約付きのProximal Operator
    return cp.clip(f, min, max)


def prox_conj(x, prox, gamma):
    return x - gamma * prox(x / gamma, 1 / gamma)


def primal_dual_solver(g, H, tau, max_iter=500, tol=1e-5):
    n = int(cp.sqrt(H.shape[1]))
    f = cp.zeros(n * n)
    f_prev = f.copy()
    gamma1 = 1e-4
    gamma2 = 1e-4

    # 勾配と発散演算子
    D = gradient_operator(n)

    # 双対変数の初期化
    y = cp.zeros(D.shape[0])

    for k in range(max_iter):
        f = prox_iota(f_prev - gamma1 * (H.T @ (H @ f_prev - g) + D.T @ y), gamma1 * tau)

        y = prox_conj(y + gamma2 * D @ (2 * f - f_prev), prox_l1, gamma2)

        # 収束判定
        norm_diff = cp.linalg.norm(f - f_prev)
        norm_f = cp.linalg.norm(f_prev)
        if norm_f == 0:
            norm_f = 1.0  # ゼロ除算を避ける
        error = norm_diff / norm_f
        print(f"iter = {k+1}: error = {error}")
        if error < tol:
            break

    return f


# %%
# DATA_PATH = '../data'
DATA_PATH = "../../OneDrive - m.titech.ac.jp/Lab/data"
OBJ_NAME = "Cameraman"
H_SETTING = "gf"
# H_SETTING = "int_p-5_lmd-100_to-True"
# H_SETTING = "p-5_lmd-100_to-False"
CAP_DATE = "241114"
EXP_DATE = "241202"
# 画像サイズ
n = 128
m = 255

# %%
# Ground Truth
f_true = cp.asarray(Image.open(f"{DATA_PATH}/sample_image{n}/{OBJ_NAME}.png").convert("L"))

# システム行列 H
loaded = cp.load(f"{DATA_PATH}/{EXP_DATE}/systemMatrix/H_matrix_{H_SETTING}.npz")
H = csp.csr_matrix(
    (cp.array(loaded["data"]), cp.array(loaded["indices"]), cp.array(loaded["indptr"])), shape=tuple(loaded["shape"])
)
print(H.shape)

# 観測画像 g
captured = cp.asarray(Image.open(f"{DATA_PATH}/capture_{CAP_DATE}/{OBJ_NAME}.png").convert("L"))
black = myUtil.calculate_bias(m**2, DATA_PATH, CAP_DATE)
g = captured.ravel() - black

# %%
# 正則化パラメータ
tau_reg = 0.1

# 最適化問題を解く
f_reconstructed = primal_dual_solver(g, H, tau_reg)

# 結果の表示
f_reconstructed_image = f_reconstructed.reshape((n, n))

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Ground Truth")
plt.imshow(cp.asnumpy(f_true), cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Captured Image")
plt.imshow(cp.asnumpy(g.reshape((m, m))), cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Reconstructed Image")
plt.imshow(cp.asnumpy(f_reconstructed_image), cmap="gray")
plt.axis("off")

plt.show()

# %%
f_image = Image.fromarray((f_reconstructed_image * 255).astype(np.uint8), mode="L")
# display(f_image)

if not os.path.exists(f"{DATA_PATH}/{EXP_DATE}/reconst"):
    os.makedirs(f"{DATA_PATH}/{EXP_DATE}/reconst")
SAVE_PATH = f"{DATA_PATH}/{EXP_DATE}/reconst/{OBJ_NAME}_{H_SETTING}_primal_t-{tau_reg}.png"
f_image.save(SAVE_PATH, format="PNG")
print(SAVE_PATH)
