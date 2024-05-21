import numpy as np
import matplotlib.pyplot as plt
import localConfig
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

H_tensor = np.load(localConfig.DATA_PATH + '/systemMatrix/H_matrix_tensor.npy')
H_tensor = H_tensor[:, :, 31, :]

H_tensor[H_tensor <= 1e-3] = 0

nonzero_indices = np.nonzero(H_tensor)
X_sr = np.column_stack(nonzero_indices)  # 事例
t_sr = H_tensor[nonzero_indices]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(projection='3d')
ax.view_init(30, -60)
ax.scatter(X_sr.T[0], X_sr.T[1], X_sr.T[2], c=t_sr, s=50)
ax.set_xlim(0, 127)
ax.set_ylim(0, 127)
ax.set_zlim(0, 63)
plt.show()

pca = PCA(n_components=3)
pca.fit(X_sr)
images_map = pca.transform(X_sr)

fig = plt.figure()
plt.scatter(images_map.T[0], images_map.T[1], c=t_sr)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

pc1 = pca.components_[0]
pc2 = pca.components_[1]

# 法線ベクトルの計算
normal_vector = np.cross(pc1, pc2)

# 平面上の点の選択
mean_scores = np.mean(images_map, axis=0)
point_on_plane = pca.inverse_transform(mean_scores)

# 回帰平面の方程式
# ax + by + cz + d = 0
a, b, c = normal_vector
d = -np.dot(normal_vector, point_on_plane)

print("回帰平面の方程式:")
print(f"{a}x + {b}y + {c}z + {d} = 0")

# 回帰平面のプロット用データの生成
x_min, x_max = np.min(X_sr[:, 0]), np.max(X_sr[:, 0])
y_min, y_max = np.min(X_sr[:, 1]), np.max(X_sr[:, 1])
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
zz = (-a * xx - b * yy - d) / c

# プロットの作成
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 回帰平面のプロット
ax.plot_surface(xx, yy, zz, alpha=0.5)

# ラベルの設定
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(0, 127)
ax.set_ylim(0, 127)
ax.set_zlim(0, 63)

# タイトルの設定
ax.set_title('PCA Regression Plane')

plt.show()
