import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import package.localConfig as localConfig


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
    values = []

    for i in range(d):
        for j in range(h):
            for k in range(w):
                if tensor[i, j, k] >= threshold:
                    x_coords.append(i)
                    y_coords.append(j)
                    z_coords.append(k)
                    values.append(tensor[i, j, k])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 値の範囲によって色を設定
    cmap = cm.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(values)))

    ax.scatter(x_coords, y_coords, z_coords, c=colors, cmap=cmap)

    ax.set_xlabel('i')
    ax.set_ylabel('j')
    ax.set_zlabel('l')
    ax.set_title(f'Tensor Values >= {threshold}')

    ax.set_xlim(0, d)
    ax.set_ylim(0, h)
    ax.set_zlim(0, w)

    # カラーバーを追加
    fig.colorbar(ax.scatter(x_coords, y_coords, z_coords,
                 c=values, cmap=cmap), ax=ax, label='Value')

    if save_path:
        plt.savefig(save_path)
    plt.show()


INDEX = 31

H_tensor = np.load(localConfig.DATA_PATH + '/systemMatrix/H_matrix_tensor.npy')
THRESHOLD = 0.005  # 表示する値の閾値
SAVE_PATH = localConfig.DATA_PATH + '/240331/system_tensor_ijl_k' + str(INDEX)

plot_tensor(H_tensor[:, :, INDEX, :], THRESHOLD)
