"""Generate system matrix from mono images"""

import os
import re
import numpy as np
from PIL import Image


class SystemMatrixMono:
    def __init__(self, data_path, pattern_name):
        self.data_path = data_path
        self.pattern_name = pattern_name
        self.n = 64

    def load_images(self, folder_path):
        files = os.listdir(folder_path)
        files.sort(key=lambda f: int(re.search(rf"{self.pattern_name}_(\d+).png", f).group(1)))
        images = []

        for file in files:
            one = np.ones((self.n, self.n))
            img = np.asarray(Image.open(os.path.join(folder_path, file)))
            img_vec = (2 * img - one).flatten()
            images.append(img_vec)
        return np.column_stack(images)

    def generate(self):
        F = self.load_images(f"{self.data_path}/{self.pattern_name}{self.n}_input/")
        G = self.load_images(f"{self.data_path}/{self.pattern_name}{self.n}_cap_240814/")
        H = (G @ F.T) / (self.n**2)

        return H
