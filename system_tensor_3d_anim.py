"""Visualize system tensor 3D"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import localConfig


def plot_tensor(tensor, threshold, index):
    """
    Plot the 3D points of a tensor that are greater than or equal to a threshold value.

    Args:
        tensor (numpy.ndarray): The input tensor of shape (d, h, w).
        threshold (float): The threshold value for selecting points to plot.
        index (int): The index of the current frame in the animation.

    Returns:
        None
    """
    # テンソルのサイズを取得
    d, h, w = tensor.shape

    # プロットする座標を格所するリスト
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

    ax.clear()
    ax.scatter(x_coords, y_coords, z_coords)

    ax.set_xlabel('i')
    ax.set_ylabel('j')
    ax.set_zlabel('l')
    ax.set_title(f'Tensor Values >= {threshold} (k={index})')

    ax.set_xlim(0, d)
    ax.set_ylim(0, h)
    ax.set_zlim(0, w)


H_tensor = np.load(localConfig.DATA_PATH + '/systemMatrix/H_matrix_tensor.npy')
THRESHOLD = 1.0e-3  # 表示する値の閾値

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')


def update(index):
    """
    Update function for the animation.

    Args:
        index (int): The index of the current frame in the animation.

    Returns:
        None
    """
    plot_tensor(H_tensor[:, :, index, :], THRESHOLD, index)


ani = FuncAnimation(fig, update, frames=range(64), interval=200)

ani.save(localConfig.DATA_PATH + '/240331/system_tensor_ijl_1e-3.gif', writer='pillow')
plt.show()
