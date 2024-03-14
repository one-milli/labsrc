import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

n = 64
m = 128

H_tensor = np.load('../data/systemMatrix/H_matrix_tensor.npy')
print(H_tensor.shape)


# アニメーションを作成
INDEX = 95
fig, ax = plt.subplots()
imgs = []
for i in range(n):
    img = ax.imshow(H_tensor[INDEX, :, i, :], animated=True, cmap='viridis')
    title = ax.text(0.5, 1.01, f"Slice {i}", ha="center", va="bottom",
                    transform=ax.transAxes, fontsize=12)
    imgs.append([img, title])

ani = animation.ArtistAnimation(fig, imgs, interval=100, blit=True, repeat_delay=1000)
fig.colorbar(img)
plt.show()

# アニメーションをGIFとして保存
ani.save("../data/240312/tensorH_anim_k_i" + str(INDEX) + ".gif", writer="pillow")
