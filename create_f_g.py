"""create F, G"""

# pylint: disable=invalid-name
# pylint: disable=line-too-long

import os
import cupy as cp
from PIL import Image
from package import myUtil

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cap_dates = {128: "241114", 256: "241205"}
n = 256
LAMBDA = 100
RATIO = 0.05
DATA_PATH = "../data"
# DATA_PATH = "../../OneDrive - m.titech.ac.jp/Lab/data"
IMG_NAME = "hadamard"
CAP_DATE = cap_dates[n]
SETTING = f"{n}_p-{int(100*RATIO)}_lmd-{LAMBDA}"

# use_list = myUtil.get_use_list(n**2, RATIO)
use_list = myUtil.read_use_list(f"use_list{n}_5.0.mat")

G = myUtil.images2matrix(f"{DATA_PATH}/{IMG_NAME}{n}_cap_{CAP_DATE}/", use_list, n).astype(cp.float32)
F = myUtil.images2matrix(f"{DATA_PATH}/{IMG_NAME}{n}_input/", use_list, n).astype(cp.int8)
M, K = G.shape
N, K = F.shape
print("G shape:", G.shape, "F shape:", F.shape, "M=", M, "N=", N, "K=", K)
print("G max:", G.max(), "G min:", G.min(), "F max:", F.max(), "F min:", F.min())

black = myUtil.calculate_bias(M, DATA_PATH, CAP_DATE)
B = cp.tile(black[:, None], K)

G = G - B

white_img = Image.open(f"{DATA_PATH}/{IMG_NAME}{n}_cap_{CAP_DATE}/hadamard_1.png").convert("L")
white = (cp.asarray(white_img) / 255).astype(cp.float32)
white = white.ravel() - black
H1 = cp.tile(white[:, None], K)

F_hat = 2 * F - 1
G_hat = 2 * G - H1
del F, G, H1

cp.save(f"{DATA_PATH}/capture_{CAP_DATE}/F_hat.npy", F_hat)
cp.save(f"{DATA_PATH}/capture_{CAP_DATE}/G_hat.npy", G_hat)
