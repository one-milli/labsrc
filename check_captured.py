import os
import numpy as np
import cv2
from matplotlib import pyplot as plt


def calculate_dot_products(directory_path):
    image_files = sorted([f for f in os.listdir(
        directory_path) if f.endswith(('.png'))])
    dot_products = []

    for i in range(len(image_files) - 1):
        img1 = cv2.imread(os.path.join(directory_path, image_files[i]))
        img2 = cv2.imread(os.path.join(directory_path, image_files[i + 1]))

        # 必要に応じて画像のサイズや色空間を調整
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # グレースケール化
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # 画像を1次元配列に変換して内積計算
        dot_product = np.dot(img1.flatten(), img2.flatten())
        dot_products.append(dot_product)

    return dot_products


def plot_dot_products(dot_products):
    plt.figure(figsize=(10, 6))  # グラフサイズを指定
    plt.bar(range(len(dot_products)), dot_products)
    plt.xlabel('Image Pair Index', fontsize=12)
    plt.ylabel('Dot Product', fontsize=12)
    plt.title('Dot Products of Adjacent Images', fontsize=14)
    plt.grid(axis='y', linestyle='--')
    plt.show()


directory = '~/data/hadamard64_cap_R_230516_128'  # 画像ディレクトリへのパス
dot_products = calculate_dot_products(directory)
plot_dot_products(dot_products)
