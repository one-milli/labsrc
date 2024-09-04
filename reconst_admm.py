# %%
import numpy as np
import scipy.sparse as ssp
import scipy.io as sio
from PIL import Image
import admm

# %%
# DATA_PATH = '../../../OneDrive - m.titech.ac.jp/Lab/data'
DATA_PATH = "../data"
OBJ_NAME = "Cameraman"
H_SETTING = "hadamard_pr-du_p-5_lmd1-10.0_lmd2-1000.0"
# H_SETTING = "gf"
n = 128
m = 192
tau = 1e0


# %%
def createDrgb(n):
    I = ssp.eye(n**2, format="lil")

    Dx = I - ssp.lil_matrix(np.roll(I.toarray(), 1, axis=1))
    Dx[n - 1 :: n, :] = 0
    Dy = I - ssp.lil_matrix(np.roll(I.toarray(), n, axis=1))
    Dy[-n:, :] = 0
    D0 = ssp.lil_matrix((n**2, n**2))

    D = ssp.block_array([[Dx, D0, D0], [D0, Dx, D0], [D0, D0, Dx], [Dy, D0, D0], [D0, Dy, D0], [D0, D0, Dy]])

    return D


def createDmono(n):
    I = ssp.eye(n**2, format="lil")

    Dx = I - ssp.lil_matrix(np.roll(I.toarray(), 1, axis=1))
    Dx[n - 1 :: n, :] = 0
    Dy = I - ssp.lil_matrix(np.roll(I.toarray(), n, axis=1))
    Dy[-n:, :] = 0

    D = ssp.vstack([Dx, Dy])

    return D


# D = createDrgb(n)
D = createDmono(n)
D = D.toarray()
print("Created D")

# %%
captured = Image.open(f"{DATA_PATH}/capture_240814/{OBJ_NAME}.png").convert("L")
captured = captured.resize((m, m))
captured = np.array(captured)
g = captured.reshape(-1, 1)

# %%
H = np.load(f"{DATA_PATH}/240825/systemMatrix/H_matrix_{H_SETTING}.npy")
print("H shape:", H.shape, "type(H):", type(H), "H.dtype:", H.dtype)

# %%
admm = admm.Admm(H, g, D, tau)
f, err = admm.solve()

# %%
f = np.clip(f, 0, 1) * 255
F = f.reshape(n, n)
F = F.astype(np.uint8)
F_image = Image.fromarray(F)
mu1 = np.log10(admm.mu1)
mu2 = np.log10(admm.mu2)
mu3 = np.log10(admm.mu3)
SAVE_PATH = f"{DATA_PATH}/240825/reconst/{OBJ_NAME}_{H_SETTING}_admm_t-{tau}_m{mu1}m{mu2}m{mu3}.png"
F_image.save(SAVE_PATH)
print(SAVE_PATH)
