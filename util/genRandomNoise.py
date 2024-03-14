import numpy as np
import matplotlib.pyplot as plt
import os


def generate_full_color_noise_images(num_images, size):
    images = []
    for _ in range(num_images):
        # Generating a full-color noise image with values between 0 and 1
        image = np.random.rand(size, size, 3)  # 3 for RGB channels
        images.append(image)
    return images


def save_full_color_noise_images(images, directory="color_noise_images"):
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i, image in enumerate(images):
        plt.imsave(f"{directory}/image_{i+1}.png", image)


def generate_binary_noise_images(num_images, size):
    images = []
    for _ in range(num_images):
        image = np.random.randint(0, 2, (size, size))
        images.append(image)
    return images


def save_binary_noise_images(images, directory="noise_images"):
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i, image in enumerate(images):
        plt.imsave(f"{directory}/image_{i+1}.png", image, cmap='gray')


# 100種類のノイズ画像を生成
# noise_images = generate_binary_noise_images(100, 128)
# 生成された画像を保存
# save_binary_noise_images(noise_images, "../../data/random_noise")

# Generate 100 full-color noise images of size 128x128
color_noise_images = generate_full_color_noise_images(100, 128)
# Save the generated images
save_full_color_noise_images(
    color_noise_images, "../../data/random_color_noise")
