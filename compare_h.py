import numpy as np


def tensor_to_matrix(tensor, m, n):
    return tensor.reshape(m * m, n * n)


DATA_PATH = "../data"

H_SETTING = "hadamard_FISTA_p-5_lmd-1_m-255"
H = np.load(f"{DATA_PATH}/241103/systemMatrix/H_matrix_{H_SETTING}.npy")

H_SETTING = "hadamard_FISTA_p-5_lmd-1_m-128"
tensor = np.load(f"{DATA_PATH}/241103/systemMatrix/H_tensor_int_{H_SETTING}.npy")
print(f"tensor shape: {tensor.shape}")
interpolated_H = tensor_to_matrix(tensor, 255, 128)

rem = np.linalg.norm(H - interpolated_H)
print(f"rem: {rem}")
