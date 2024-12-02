# %%
import os
import math
import numpy as np
import cupy as cp
import scipy as sp
import matplotlib.pyplot as plt
from IPython.display import display
from PIL import Image
import package.myUtil as myUtil

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
n = 128
DO_THIN_OUT = False
SAVE_AS_SPARSE = True
# DATA_PATH = "../../OneDrive - m.titech.ac.jp/Lab/data"
DATA_PATH = "../data"
IMG_NAME = "hadamard"
CAP_DATE = "241114"
EXP_DATE = "241202"
DIRECTORY = f"{DATA_PATH}/{EXP_DATE}"
SETTING = f"gf"

if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)
if not os.path.exists(DIRECTORY + "/systemMatrix"):
    os.makedirs(DIRECTORY + "/systemMatrix")

# %%
use_list = myUtil.get_use_list(n * n, 1.0)
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
del F, G, H1

# %%
H = G_hat @ F_hat.T / N
# H[:, 0] = H[:, 1]
print("H shape:", H.shape)

# %%
if SAVE_AS_SPARSE:
    # Thresholding
    threshold = 1e-4
    H = cp.where(cp.abs(H) < threshold, 0, H)
    H = sp.sparse.csr_matrix(H.get())
    sp.io.mmwrite(f"{DIRECTORY}/systemMatrix/H_matrix_{SETTING}.mtx", H)
    # sp.io.savemat(f"{DIRECTORY}/systemMatrix/H_matrix_{SETTING}.mat", {"H": H.get()})
    # print(f"Saved {DIRECTORY}/systemMatrix/H_matrix_{SETTING}.mat")
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
