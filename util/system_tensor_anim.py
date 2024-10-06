"""Generate animation"""

import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import localConfig


def matrix_to_tensor(H, m, n):
    H_tensor = np.zeros((m, m, n, n))
    for i in range(m):
        for j in range(m):
            for k in range(n):
                for l in range(n):
                    H_tensor[i, j, k, l] = H[i * m + j, k * n + l]
    return H_tensor


H_matrix = np.load(f"{localConfig.DATA_PATH}/systemMatrix/H_matrix_FISTA_hadamard_50.0p.npy")

n = 64
m = 128

# H_tensor = np.load(localConfig.DATA_PATH + '/systemMatrix/H_matrix_tensor.npy')
H_tensor = matrix_to_tensor(H_matrix, m, n)
print(H_tensor.shape)

INDEX = 63
fig, ax = plt.subplots()
imgs = []

vmin = np.min(H_tensor[INDEX, :, :, :])  # change
vmax = np.max(H_tensor[INDEX, :, :, :])  # change
ax.set_ylabel('k')
ax.set_xlabel('l')

for i in range(m):
    img = ax.imshow(H_tensor[INDEX, i, :, :], animated=True, cmap='viridis', vmin=vmin, vmax=vmax)
    title = ax.text(0.5, 1.01, f"i={INDEX}, j={i}", ha="center", va="bottom",
                    transform=ax.transAxes, fontsize=12)
    imgs.append([img, title])

ani = animation.ArtistAnimation(
    fig, imgs, interval=100, blit=True, repeat_delay=1000)
fig.colorbar(img)
plt.show()

ani.save(localConfig.DATA_PATH + "/240509/tensorH_50p_anim_j_i" + str(INDEX) + ".gif", writer="pillow")
