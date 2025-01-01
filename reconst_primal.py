# %%
import os
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as csp
import matplotlib.pyplot as plt
import package.myUtil as myUtil
from PIL import Image
from itertools import product


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


def prox_l12(f, gamma):
    # 混合L1,2ノルムのProximal Operator
    factor = cp.linalg.norm(f.reshape((-1, 2), order="F"), axis=1)
    factor = cp.where(factor > 0, cp.maximum(factor - gamma, 0) / factor, 0)
    factor = cp.tile(factor, 2)
    return factor * f


def prox_iota(f, mini=0, maxi=1):
    # 指示関数のProximal Operator
    return cp.clip(f, mini, maxi)
    # return f


def prox_conj(x, prox, gamma):
    return x - gamma * prox(x / gamma, 1 / gamma)


def primal_dual_solver(g, H, tau, gamma1, gamma2, max_iter=10000, tol=1e-4):
    n = int(cp.sqrt(H.shape[1]))
    f = cp.zeros(n * n)
    f_prev = cp.zeros(n * n) / 2

    D = gradient_operator(n)

    y = cp.zeros(D.shape[0])

    for k in range(max_iter):
        f = prox_iota(f_prev - gamma1 * (H.T @ (H @ f_prev - g) + D.T @ y))

        y = prox_conj(y + gamma2 * D @ (2 * f - f_prev), prox_l12, gamma2 * tau)

        # 収束判定
        norm_diff = cp.linalg.norm(f - f_prev)
        norm_f = cp.linalg.norm(f_prev)
        if norm_f == 0:
            norm_f = 1.0  # ゼロ除算を避ける
        error = norm_diff / norm_f
        f_prev = f
        if (k + 1) % 100 == 0 or error < tol:
            print(f"iter = {k+1}: error = {error}")
        if error < tol:
            break

    return f


# %%
# 画像サイズ
n = 256
m = 550
DATA_PATH = "../data"
# DATA_PATH = "../../OneDrive - m.titech.ac.jp/Lab/data"
OBJ_NAME = "Cameraman"
# H_SETTING = "gf"
H_SETTING = f"{n}_p-5_lmd-100"
CAP_DATE = "241216"
EXP_DATE = "241230"

# %%
# システム行列 H
loaded = cp.load(f"{DATA_PATH}/{EXP_DATE}/systemMatrix/H_matrix_{H_SETTING}.npz")
H = csp.csr_matrix(
    (cp.array(loaded["data"]), cp.array(loaded["indices"]), cp.array(loaded["indptr"])),
    shape=tuple(loaded["shape"]),
)
print(f"shape: {H.shape}, nnz: {H.nnz}({H.nnz / H.shape[0] / H.shape[1] * 100:.2f}%)")
# myUtil.plot_sparse_matrix_cupy(H, row_range=(5500, 6000), col_range=(4500, 5000), markersize=1)

# 観測画像 g
captured = (cp.asarray(Image.open(f"{DATA_PATH}/capture_{CAP_DATE}/{OBJ_NAME}.png").convert("L")) / 255).astype(
    cp.float32
)
bias = myUtil.calculate_bias(DATA_PATH, CAP_DATE)
g = captured.ravel() - bias

# %%
# 正則化パラメータ
TAU = 1e2

# GAMMA1とGAMMA2の値のリストを作成
gamma_values = [10**i for i in range(-6, 0)]

# 出力ディレクトリの作成
output_dir = f"{DATA_PATH}/{EXP_DATE}/reconst"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# %%
# GAMMA1とGAMMA2の全組み合わせをループ
for GAMMA1, GAMMA2 in product(gamma_values, repeat=2):
    print(f"Processing GAMMA1={GAMMA1}, GAMMA2={GAMMA2}")

    # 最適化問題を解く
    f_reconstructed = primal_dual_solver(g, H, TAU, GAMMA1, GAMMA2)

    # 結果の表示
    f_reconstructed_image = cp.asnumpy(f_reconstructed.reshape((n, n)))

    # Ground Truth
    f_true = np.asarray(Image.open(f"{DATA_PATH}/sample_image{n}/{OBJ_NAME}.png").convert("L"))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Ground Truth")
    plt.imshow(f_true, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Captured Image")
    plt.imshow(cp.asnumpy(g.reshape((m, m))), cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"Reconstructed Image\nGAMMA1={GAMMA1}, GAMMA2={GAMMA2}")
    plt.imshow(f_reconstructed_image, cmap="gray")
    plt.axis("off")

    plt.tight_layout()

    # 保存用のファイル名を作成
    gamma1_log = int(np.log10(GAMMA1))
    gamma2_log = int(np.log10(GAMMA2))
    SAVE_PATH = (
        f"{output_dir}/{OBJ_NAME}_{H_SETTING}_primal_tau-{int(np.log10(TAU))}_" f"g1-{gamma1_log}_g2-{gamma2_log}.png"
    )

    # 画像を保存
    plt.savefig(SAVE_PATH, format="PNG")
    plt.close()

    print(f"Saved reconstructed image to {SAVE_PATH}\n")

# %%
print("全ての組み合わせの処理が完了しました。")
