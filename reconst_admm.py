 %%
import numpy as np
import scipy.sparse as sps
import scipy.io as sio
from PIL import Image
import admm

# %%
# DATA_PATH = '../../../OneDrive - m.titech.ac.jp/Lab/data'
DATA_PATH = '../data'
OBJ_NAME = "Text"
H_SETTING = "hadamard_FISTA_l122_p-10_lmd-100"
n = 128
m = 256
tau = 1e1

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
    I = sps.eye(n**2, format="lil")

    Dx = I - sps.lil_matrix(np.roll(I.toarray(), 1, axis=1))
    Dx[n - 1 :: n, :] = 0
    Dy = I - sps.lil_matrix(np.roll(I.toarray(), n, axis=1))
    Dy[-n:, :] = 0

    D = sps.vstack([Dx, Dy])

    return D


# D = createDrgb(n)
D = createDmono(n)
print("Created D")

# %%
# captured = Image.open(f"{DATA_PATH}/capture_240814/{OBJ_NAME}_edited.png")
captured = Image.open(f"{DATA_PATH}/capture_240814/{OBJ_NAME}_edited.png").convert("L")
# captured = captured.crop((400, 460, 860, 920)).resize((n, n))
captured = np.array(captured)
g = captured.reshape(-1, 1)

# %%
# H = np.load(f"{DATA_PATH}/systemMatrix/H_matrix_{SETTING}.npy")
H = sio.mmread(f"{DATA_PATH}/240818/systemMatrix/H_sparse_{H_SETTING}.mtx").tocsr()
print(sio.mminfo(f"{DATA_PATH}/240818/systemMatrix/H_sparse_{H_SETTING}.mtx"))

# %%
H = H.toarray()
D = D.toarray()

# %%
admm = admm.Admm(H, g, D, tau)
f, err = admm.solve()

# %%
f = np.clip(f, 0, 1) * 255
F = f.reshape(n, n)
F = F.astype(np.uint8)
F_image = Image.fromarray(F)
F_image.save(f"{DATA_PATH}/240818/reconst/{OBJ_NAME}_admm_t-{tau}.png")
print(f"Saved {DATA_PATH}/240818/reconst/{OBJ_NAME}_admm_t-{tau}.png")

