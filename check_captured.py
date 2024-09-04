import os
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt


def cv2pil(image):
    """OpenCV型 -> PIL型"""
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


def load_images(directory):
    images = []
    for i in range(4096):
        filename = os.path.join(directory, f"hadamard_{i+1}.png")
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  # グレースケールで読み込み
        if i == 32 or i == 31:
            img2 = cv2pil(img)
            img2.show()
        images.append(img.flatten())  # 1次元配列に変換
    return np.array(images)


def calculate_inner_products(images):
    inner_products = []
    for i in range(len(images) - 1):
        inner_product = np.dot(images[i], images[i + 1])
        inner_products.append(inner_product)
    return inner_products


def plot_inner_products(inner_products):
    plt.figure(figsize=(15, 6))  # グラフサイズを調整
    plt.bar(range(len(inner_products)), inner_products)
    plt.xlabel("Image Pair Index")
    plt.ylabel("Inner Product")
    plt.title("Inner Products of Adjacent Images")
    plt.show()


directory = "../data/hadamard64_cap_R_230516_128"  # 画像のあるディレクトリを指定

images = load_images(directory)
inner_products = calculate_inner_products(images)
plot_inner_products(inner_products)
