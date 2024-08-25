"""Generate system matrix from RGB images"""
import os
import re
import numpy as np
from PIL import Image


class SystemMatrixRgb:
    def __init__(self, data_path, pattern_name):
        self.data_path = data_path
        self.pattern_name = pattern_name
        self.n = 64

    def load_images(self, folder_path, convert_gray=False):
        files = os.listdir(folder_path)
        files.sort(key=lambda f: int(re.search(fr"{self.pattern_name}_(\d+).png", f).group(1)))
        images = []
        images_r = []
        images_g = []
        images_b = []

        if convert_gray:
            for file in files:
                one = np.ones((self.n, self.n))
                img = np.asarray(Image.open(os.path.join(folder_path, file)))
                img_vec = (2 * img - one).flatten()
                images.append(img_vec)
            return np.column_stack(images)
        else:
            white = np.asarray(Image.open(os.path.join(folder_path, 'hadamard_1.png'))) / 255
            for file in files:
                img = np.asarray(Image.open(os.path.join(folder_path, file))) / 255
                img_vec_r = 2 * img[:, :, 0] - white[:, :, 0]
                img_vec_g = 2 * img[:, :, 1] - white[:, :, 1]
                img_vec_b = 2 * img[:, :, 2] - white[:, :, 2]
                images_r.append(img_vec_r.flatten())
                images_g.append(img_vec_g.flatten())
                images_b.append(img_vec_b.flatten())
            print("Finished loading images")
            return np.column_stack(images_r), np.column_stack(images_g), np.column_stack(images_b)

    def generate(self):
        F = self.load_images(f"{self.data_path}/{self.pattern_name}{self.n}_input/", convert_gray=True)
        Grr, Grg, Grb = self.load_images(f"{self.data_path}/{self.pattern_name}{self.n}_cap_R_230516_128/")
        Ggr, Ggg, Ggb = self.load_images(f"{self.data_path}/{self.pattern_name}{self.n}_cap_G_230516_128/")
        Gbr, Gbg, Gbb = self.load_images(f"{self.data_path}/{self.pattern_name}{self.n}_cap_B_230516_128/")

        Hrr = (Grr @ F.T) / (self.n**2)
        Hrg = (Ggr @ F.T) / (self.n**2)
        Hrb = (Gbr @ F.T) / (self.n**2)
        Hgr = (Grg @ F.T) / (self.n**2)
        Hgg = (Ggg @ F.T) / (self.n**2)
        Hgb = (Gbg @ F.T) / (self.n**2)
        Hbr = (Grb @ F.T) / (self.n**2)
        Hbg = (Ggb @ F.T) / (self.n**2)
        Hbb = (Gbb @ F.T) / (self.n**2)

        H = np.block([[Hrr, Hrg, Hrb], [Hgr, Hgg, Hgb], [Hbr, Hbg, Hbb]])

        return H
