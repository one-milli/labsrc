"""Generate animation"""

import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import localConfig

n = 64
m = 128

H_tensor = np.load(localConfig.DATA_PATH + 'systemMatrix/H_matrix_tensor_10p.npy')

INDEX = 31
fig, ax = plt.subplots()
imgs = []

vmin = np.min(H_tensor[:, :, INDEX, :])
vmax = np.max(H_tensor[:, :, INDEX, :])
ax.set_ylabel('i')
ax.set_xlabel('j')

for i in range(n):
    img = ax.imshow(H_tensor[:, :, INDEX, i], animated=True, cmap='viridis', vmin=vmin, vmax=vmax)
    title = ax.text(0.5, 1.01, f"k={INDEX}, l={i}", ha="center", va="bottom",
                    transform=ax.transAxes, fontsize=12)
    imgs.append([img, title])

ani = animation.ArtistAnimation(
    fig, imgs, interval=200, blit=True, repeat_delay=1000)
fig.colorbar(img)
plt.show()

ani.save(localConfig.DATA_PATH + "240331/tensorH_10p_anim_l_k" +
         str(INDEX) + ".gif", writer="pillow")
