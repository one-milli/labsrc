import numpy as np


DATA_PATH = "../data"

H_SETTING = "hadamard_FISTA_p-5_lmd-1_m-255"
H1 = np.load(f"{DATA_PATH}/241103/systemMatrix/H_matrix_{H_SETTING}.npy")
print(f"H1: {H1.shape}")

H_SETTING = "hadamard_FISTA_p-5_lmd-1_m-128"
H2 = np.load(f"{DATA_PATH}/241103/systemMatrix/H_matrix_int_{H_SETTING}.npy")
print(f"H2: {H2.shape}")

rem = np.linalg.norm(H1 - H2)
print(f"rem: {rem}")
