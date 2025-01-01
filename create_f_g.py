"""create F, G"""

# pylint: disable=invalid-name
# pylint: disable=line-too-long

import os
import numpy as np
import cupy as cp
from PIL import Image
from package import myUtil

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
n = 256
LAMBDA = 100
RATIO = 0.05
DATA_PATH = "../../OneDrive - m.titech.ac.jp/Lab/data"
CAP_DATE = "241216"
SETTING = f"{n}_p-{int(100*RATIO)}_lmd-{LAMBDA}"

# use_list = myUtil.get_use_list(n**2, RATIO)
use_list = myUtil.read_use_list(f"use_list{n}_5.0.mat")

bias_val = myUtil.calculate_bias(DATA_PATH, CAP_DATE).astype(np.float32)
G = myUtil.images2matrix(f"{DATA_PATH}/hadamard{n}_cap_{CAP_DATE}/", use_list, n, bias_val).astype(cp.float32)
F = myUtil.images2matrix(f"{DATA_PATH}/hadamard{n}_input/", use_list, n).astype(cp.int8)
K = G.shape[1]
print("G shape:", G.shape, "F shape:", F.shape, "M=", G.shape[0], "N=", F.shape[0], "K=", K)
print("G max:", G.max(), "G min:", G.min(), "F max:", F.max(), "F min:", F.min())

white_img = Image.open(f"{DATA_PATH}/hadamard{n}_cap_{CAP_DATE}/hadamard_1.png").convert("L")
white = (cp.asarray(white_img) / 255).astype(cp.float32)
white = white.reshape(-1, 1) - bias_val

F = 2 * F - 1
G = 2 * G - white

cp.save(f"{DATA_PATH}/capture_{CAP_DATE}/F_hat.npy", F)
cp.save(f"{DATA_PATH}/capture_{CAP_DATE}/G_hat.npy", G)
