"""multiplication of system matrix and image vector"""
import os
import localConfig
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

OUTPUT_SIZE = 128
DIRECTORY = localConfig.DATA_PATH + '/random_noise64_cap'
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)

system_matrix = np.load(localConfig.DATA_PATH + '/systemMatrix/H_matrix_true.npy')

for i in range(1, 2):
    subject_image = Image.open(f"{localConfig.DATA_PATH}/random_noise64_input/random_noise_{i}.png")
    subject_image = subject_image.convert('L')
    subject_image = np.asarray(subject_image).flatten() / 255

    output_image = system_matrix @ subject_image
    output_image = output_image.reshape(OUTPUT_SIZE, OUTPUT_SIZE)

    FILENAME = f"random_noise_{i}.png"

    plt.imsave(f"{DIRECTORY}/{FILENAME}", output_image, cmap='gray')
