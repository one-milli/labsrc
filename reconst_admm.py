# %%
import numpy as np
import cupy as cp
import scipy.sparse as sps
import scipy.io as sio
import cupyx.scipy.sparse as csp
from PIL import Image
import admm

# %%
# DATA_PATH = '../../../OneDrive - m.titech.ac.jp/Lab/data'
DATA_PATH = "../data"
OBJ_NAME = "Cameraman"
H_SETTING = "hadamard_pr-du_p-5_lmd1-10.0_lmd2-1000.0"
# H_SETTING = "gf"
n = 128
m = 256
tau = 1e0
cp.cuda.Device(2).use()
cp.cuda.Device(3).use()


# %%
def createDrgb(n):
    I = sps.eye(n**2, format="lil")

    Dx = I - sps.lil_matrix(np.roll(I.toarray(), 1, axis=1))
    Dx[n - 1 :: n, :] = 0
    Dy = I - sps.lil_matrix(np.roll(I.toarray(), n, axis=1))
    Dy[-n:, :] = 0
    D0 = sps.lil_matrix((n**2, n**2))

    D = sps.block_array([[Dx, D0, D0], [D0, Dx, D0], [D0, D0, Dx], [Dy, D0, D0], [D0, Dy, D0], [D0, D0, Dy]])

    return D


def createDmono(n):
    I = csp.eye(n**2, format="lil")

    Dx = I - csp.lil_matrix(np.roll(I.toarray(), 1, axis=1))
    Dx[n - 1 :: n, :] = 0
    Dy = I - csp.lil_matrix(np.roll(I.toarray(), n, axis=1))
    Dy[-n:, :] = 0

    D = csp.vstack([Dx, Dy])

    return D


# D = createDrgb(n)
D = createDmono(n)
# D = D.toarray()
print("Created D")

# %%
# captured = Image.open(f"{DATA_PATH}/capture_240814/{OBJ_NAME}.png")
captured = Image.open(f"{DATA_PATH}/capture_240814/{OBJ_NAME}.png").convert("L")
# captured = captured.crop((400, 460, 860, 920)).resize((n, n))
captured = cp.asarray(captured)
g = captured.reshape(-1, 1)

# %%
H = cp.load(f"{DATA_PATH}/240825/systemMatrix/H_matrix_{H_SETTING}.npy")
print("H shape:", H.shape, "type(H):", type(H), "H.dtype:", H.dtype)
# H = sio.mmread(f"{DATA_PATH}/240825/systemMatrix/H_sparse_{H_SETTING}.mtx").tocsr()
# print(sio.mminfo(f"{DATA_PATH}/240825/systemMatrix/H_sparse_{H_SETTING}.mtx"))
# H = H.toarray()

# %%
admm = admm.Admm(H, g, D, tau)
f, err = admm.solve()

# %%
f = np.clip(f, 0, 1) * 255
F = f.reshape(n, n)
F = F.astype(np.uint8)
F_image = Image.fromarray(F)
F_image.save(f"{DATA_PATH}/240825/reconst/{OBJ_NAME}_{H_SETTING}_admm_t-{tau}.png")
print(f"Saved {DATA_PATH}/240825/reconst/{OBJ_NAME}_{H_SETTING}_admm_t-{tau}.png")
