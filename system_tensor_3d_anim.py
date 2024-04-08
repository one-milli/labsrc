"""Visualize system tensor 3D"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import localConfig


def plot_tensor(tensor, x, index):
    # テンソルのサイズを取得
    d, h, w = tensor.shape

    # プロットする座標を格所するリスト
    x_coords = []
    y_coords = []
    z_coords = []

    for i in range(d):
        for j in range(h):
            for k in range(w):
                if tensor[i, j, k] >= x:
                    x_coords.append(i)
                    y_coords.append(j)
                    z_coords.append(k)

    ax.clear()
    ax.scatter(x_coords, y_coords, z_coords)

    ax.set_xlabel('i')
    ax.set_ylabel('j')
    ax.set_zlabel('l')
    ax.set_title(f'Tensor Values >= {x} (k={index})')

    ax.set_xlim(0, d)
    ax.set_ylim(0, h)
    ax.set_zlim(0, w)


H_tensor = np.load(localConfig.DATA_PATH + 'systemMatrix/H_matrix_tensor_10p.npy')
x = 0.001  # 表示する値の閾値

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')


def update(index):
    plot_tensor(H_tensor[:, :, index, :], x, index)


ani = FuncAnimation(fig, update, frames=range(64), interval=200)

ani.save(localConfig.DATA_PATH + '240331/system_tensor_10p_ijl_0001.gif', writer='pillow')
plt.show()
