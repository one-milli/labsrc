import os
import numpy as np
import cupy as cp
import scipy.sparse as ssp
from PIL import Image
from IPython.display import display
import package.myUtil as myUtil
import admm

DATA_PATH = "../data"
OBJ_NAME = "Cameraman"
# H_SETTING = "gf"
H_SETTING = "int_p-5_lmd-100_to-True"
H_SETTING = "p-5_lmd-100_to-False"
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

captured = Image.open(f"{DATA_PATH}/capture_{CAP_DATE}/{OBJ_NAME}.png").convert("L")
captured = cp.asarray(captured)
black = myUtil.calculate_bias(m**2, DATA_PATH, CAP_DATE)
g = captured.ravel() - black

H = cp.load(f"{DATA_PATH}/{EXP_DATE}/systemMatrix/H_matrix_{H_SETTING}.npy").astype(cp.float32)
print("H shape:", H.shape, "type(H):", type(H), "H.dtype:", H.dtype)

admm = admm.Admm(H, g, D)
f, err = admm.solve()

f = cp.clip(f, 0, 1)
f = cp.asnumpy(f.reshape(n, n))
f_image = Image.fromarray((f * 255).astype(np.uint8), mode="L")
display(f_image)

tau = np.log10(admm.tau)
mu1 = np.log10(admm.mu1)
mu2 = np.log10(admm.mu2)
mu3 = np.log10(admm.mu3)

if not os.path.exists(f"{DATA_PATH}/{EXP_DATE}/reconst"):
    os.makedirs(f"{DATA_PATH}/{EXP_DATE}/reconst")
SAVE_PATH = f"{DATA_PATH}/{EXP_DATE}/reconst/{OBJ_NAME}_{H_SETTING}_admm_t-{tau}_m{mu1}m{mu2}m{mu3}.png"
f_image.save(SAVE_PATH, format="PNG")
print(SAVE_PATH)
