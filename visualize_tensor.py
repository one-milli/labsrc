import numpy as np
import matplotlib.pyplot as plt
import localConfig
import os


output_dir = localConfig.DATA_PATH + "/240509/heatmaps"
os.makedirs(output_dir, exist_ok=True)

H = np.load(localConfig.DATA_PATH + '/systemMatrix/H_matrix_tensor.npy')

heatmap_size = 2

grid_size = (8, 8)

# ヒートマップの一覧を64枚に分ける
for page in range(64):
    fig, axs = plt.subplots(grid_size[0], grid_size[1], figsize=(20, 20))
    axs = axs.ravel()

    for idx, (k, l) in enumerate(np.ndindex(64, 64)):
        if idx >= page * 64 and idx < (page + 1) * 64:
            heatmap = H[:, :, k, l]
            ax = axs[idx % 64]
            im = ax.imshow(heatmap, cmap='viridis', vmin=0, vmax=1)
            ax.set_title(f"k={k}, l={l}")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.suptitle(f"Heatmaps of ij-slices (Page {page+1})", fontsize=24)
    plt.subplots_adjust(top=0.95)
    cbar = fig.colorbar(im, ax=axs.tolist(), shrink=0.6)
    # plt.show()
    plt.savefig(os.path.join(output_dir, f"heatmaps_page_{page+1}.png"))
    plt.close(fig)
