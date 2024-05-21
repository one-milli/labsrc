"""Generate random noise images and save them to disk."""
import os
import localConfig
import numpy as np
import matplotlib.pyplot as plt


def generate_full_color_noise_images(num_images, size):
    images = []
    for _ in range(num_images):
        image = np.random.rand(size, size, 3)  # 3 for RGB channels
        images.append(image)
    return images


def save_full_color_noise_images(images, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i, image in enumerate(images):
        plt.imsave(f"{directory}/random_noise_{i+1}.png", image)


def generate_binary_noise_images(num_images, size):
    images = []
    for _ in range(num_images):
        image = np.random.randint(0, 2, (size, size))
        images.append(image)
    return images


def save_binary_noise_images(images, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i, image in enumerate(images):
        plt.imsave(f"{directory}/random_noise_{i+1}.png", image, cmap='gray')


NUM_IMAGES = 4096
IMAGE_SIZE = 64
IMAGE_PATH = localConfig.DATA_PATH + "/random_noise" + str(IMAGE_SIZE) + "_input"

noise_images = generate_binary_noise_images(NUM_IMAGES, IMAGE_SIZE)
save_binary_noise_images(noise_images, IMAGE_PATH)

# color_noise_images = generate_full_color_noise_images(NUM_IMAGES, IMAGE_SIZE)
# save_full_color_noise_images(color_noise_images, IMAGE_PATH)
