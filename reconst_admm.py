import os
import numpy as np
import cupy as cp
import scipy.sparse as ssp
from PIL import Image
from package import myUtil
import admm

DATA_PATH = "../data"

# OBJ_NAMES = ["White", "Cameraman", "Text", "Daruma", "Woman"]
OBJ_NAMES = ["Woman"]

H_SETTINGS = ["gf", "p-5_lmd-100_to-False", "int_p-5_lmd-100_to-True"]

CAP_DATE = "241114"
EXP_DATE = "241127"
n = 128
m = 255


def create_D_mono(n):
    I = ssp.eye(n**2, format="csr")

    Dx = I - ssp.csr_matrix(np.roll(I.toarray(), 1, axis=1))
    Dx[n - 1 :: n, :] = 0
    Dy = I - ssp.csr_matrix(np.roll(I.toarray(), n, axis=1))
    Dy[-n:, :] = 0

    return ssp.vstack([Dx, Dy])


D = create_D_mono(n)
D = cp.array(D.toarray())

# すべてのOBJ_NAMEとH_SETTINGの組み合わせを処理
for OBJ_NAME in OBJ_NAMES:
    captured_path = f"{DATA_PATH}/capture_{CAP_DATE}/{OBJ_NAME}.png"

    captured = Image.open(captured_path).convert("L")
    captured = cp.asarray(captured)
    black = myUtil.calculate_bias(m**2, DATA_PATH, CAP_DATE)
    g = captured.ravel() - black

    for H_SETTING in H_SETTINGS:
        H_matrix_path = f"{DATA_PATH}/{EXP_DATE}/systemMatrix/H_matrix_{H_SETTING}.npy"

        H = cp.load(H_matrix_path).astype(cp.float32)
        print(f"Processing OBJ_NAME: {OBJ_NAME}, H_SETTING: {H_SETTING}")
        print("H shape:", H.shape, "type(H):", type(H), "H.dtype:", H.dtype)

        admm_solver = admm.Admm(H, g, D)
        f, err = admm_solver.solve()

        f = cp.clip(f, 0, 1)
        f = cp.asnumpy(f.reshape(n, n))
        f_image = Image.fromarray((f * 255).astype(np.uint8), mode="L")

        tau = np.log10(admm_solver.tau)
        mu1 = np.log10(admm_solver.mu1)
        mu2 = np.log10(admm_solver.mu2)
        mu3 = np.log10(admm_solver.mu3)

        reconst_dir = f"{DATA_PATH}/{EXP_DATE}/reconst"
        if not os.path.exists(reconst_dir):
            os.makedirs(reconst_dir)

        SAVE_PATH = f"{reconst_dir}/{OBJ_NAME}_{H_SETTING}_admm_t-{tau:.2f}_m{mu1:.2f}m{mu2:.2f}m{mu3:.2f}.png"
        f_image.save(SAVE_PATH, format="PNG")
        print(f"Saved: {SAVE_PATH}")
