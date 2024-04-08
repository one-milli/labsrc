"""Visualize system tensor 3D"""

import numpy as np
import matplotlib.pyplot as plt
import localConfig


def plot_tensor(tensor, threshold, save_path=None):
    """
    Plot the 3D points of a tensor that are greater than or equal to a threshold value.

    Args:
        tensor (numpy.ndarray): The input tensor of shape (d, h, w).
        threshold (float): The threshold value for selecting points to plot.
        save_path (str, optional): The path to save the plot image. If None, the plot is not saved.

    Returns:
        None
    """
    # テンソルのサイズを取得
    d, h, w = tensor.shape

    # プロットする座標を格納するリスト
    x_coords = []
    y_coords = []
    z_coords = []

    for i in range(d):
        for j in range(h):
            for k in range(w):
                if tensor[i, j, k] >= threshold:
                    x_coords.append(i)
                    y_coords.append(j)
                    z_coords.append(k)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_coords, y_coords, z_coords)

    ax.set_xlabel('i')
    ax.set_ylabel('j')
    ax.set_zlabel('l')
    ax.set_title(f'Tensor Values >= {threshold}')

    ax.set_xlim(0, d)
    ax.set_ylim(0, h)
    ax.set_zlim(0, w)

    if save_path:
        plt.savefig(save_path)
    plt.show()


INDEX = 1

H_tensor = np.load(localConfig.DATA_PATH + 'systemMatrix/H_matrix_tensor.npy')
THRESHOLD = 0.005  # 表示する値の閾値
SAVE_PATH = localConfig.DATA_PATH + '240331/system_tensor_ijl_k' + str(INDEX)

plot_tensor(H_tensor[:, :, INDEX, :], THRESHOLD, SAVE_PATH)
