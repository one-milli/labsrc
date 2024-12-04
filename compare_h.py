import cupy as cp

DATA_PATH = "../data"

H1_SETTING = "gf"
H1 = cp.load(f"{DATA_PATH}/241127/systemMatrix/H_matrix_{H1_SETTING}.npy")
print(f"H1: {H1.shape}")

H2_SETTING = "int_p-5_lmd-100_to-True"
# H2 = cp.load(f"{DATA_PATH}/241127/systemMatrix/H_matrix_{H2_SETTING}.npy")
# print(f"H2: {H2.shape}")

H2=cp.zeros(H1.shape)
rem = cp.linalg.norm(H1 - H2)
print(f"rem: {rem}")
