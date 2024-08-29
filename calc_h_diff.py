import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "../data/240825"

rem = []

H_gf = np.load(f"{DATA_PATH}/systemMatrix/H_matrix_gf.npy")
H50 = np.load(f"{DATA_PATH}/systemMatrix/H_matrix_hadamard_FISTA_l122_p-50_lmd-100.npy")
rem50 = np.linalg.norm(H_gf - H50, "fro")
rem.append(rem50)
del H50

H25 = np.load(f"{DATA_PATH}/systemMatrix/H_matrix_hadamard_FISTA_l122_p-25_lmd-100.npy")
rem25 = np.linalg.norm(H_gf - H25, "fro")
rem.append(rem25)
del H25

H10 = np.load(f"{DATA_PATH}/systemMatrix/H_matrix_hadamard_FISTA_l122_p-10_lmd-100.npy")
rem10 = np.linalg.norm(H_gf - H10, "fro")
rem.append(rem10)
del H10

H5 = np.load(f"{DATA_PATH}/systemMatrix/H_matrix_hadamard_FISTA_l122_p-5_lmd-100.npy")
rem5 = np.linalg.norm(H_gf - H5, "fro")
rem.append(rem5)
del H5

H1 = np.load(f"{DATA_PATH}/systemMatrix/H_matrix_hadamard_FISTA_l122_p-1_lmd-100.npy")
rem1 = np.linalg.norm(H_gf - H1, "fro")
rem.append(rem1)
del H1

# plot
x = [8192, 4096, 1638, 819, 163]
plt.plot(x, rem)
plt.xlabel("K value")
plt.ylabel("Frobenius norm")
plt.xticks(x, x)
plt.savefig(f"{DATA_PATH}/h_diff.png")
