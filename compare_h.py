import cupy as cp

DATA_PATH = "../data"

H1_SETTING = "hadamard_FISTA_p-5_lmd-1_m-255"
H1 = cp.load(f"{DATA_PATH}/241103/systemMatrix/H_matrix_{H1_SETTING}.npy")
print(f"H1: {H1.shape}")

H2_SETTING = "hadamard_FISTA_p-5_lmd-1_m-128"
H2 = cp.load(f"{DATA_PATH}/241103/systemMatrix/H_matrix_int_{H2_SETTING}.npy")
print(f"H2: {H2.shape}")

rem = cp.linalg.norm(H1 - H2)
print(f"rem: {rem}")
