"""Generate system matrix from mono images analytically"""

import os
import re
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from PIL import Image


class SystemMatrixMono:
    def __init__(self, data_path, pattern_name):
        self.data_path = data_path
        self.pattern_name = pattern_name
        self.n = 128
        self.m = 192

    def load_images(self, folder_path, is_f=False):
        files = os.listdir(folder_path)
        files.sort(key=lambda f: int(re.search(rf"{self.pattern_name}_(\d+).png", f).group(1)))
        images = []

        if is_f:
            for file in files:
                one = np.ones((self.n, self.n))
                img = np.asarray(Image.open(os.path.join(folder_path, file)))
                img_vec = (2 * img - one).flatten()
                images.append(img_vec)
        else:
            white = (
                np.asarray(
                    Image.open(os.path.join(folder_path, "hadamard_1.png")).convert("L").resize((self.m, self.m))
                )
                / 255
            )
            for file in files:
                img = (
                    np.asarray(Image.open(os.path.join(folder_path, file)).convert("L").resize((self.m, self.m))) / 255
                )
                img_vec = (2 * img - white).flatten()
                images.append(img_vec)
        return np.column_stack(images)

    def generate(self):
        F = self.load_images(f"{self.data_path}/{self.pattern_name}{self.n}_input/", is_f=True).astype(np.int8)
        print("F shape:", F.shape)
        G = self.load_images(f"{self.data_path}/{self.pattern_name}{self.n}_cap_240814/").astype(np.float32)
        print("G shape:", G.shape)
        F_gpu = cp.asarray(F)
        G_gpu = cp.asarray(G)

        res_gpu = ((G_gpu @ F_gpu.T) / (self.n**2)).astype(np.float32)
        res = cp.asnumpy(res_gpu)

        return res


if __name__ == "__main__":
    DATA_PATH = "../data"
    PATTERN_NAME = "hadamard"

    sm = SystemMatrixMono(DATA_PATH, PATTERN_NAME)
    H = sm.generate()
    np.save(f"{DATA_PATH}/240825/systemMatrix/H_matrix_gf.npy", H)

    SAMPLE_NAME = "Cameraman"
    sample_image = Image.open(f"{DATA_PATH}/sample_image{sm.n}/{SAMPLE_NAME}.png").convert("L")
    sample_image = np.asarray(sample_image).flatten() / 255

    Hf = H @ sample_image
    Hf_img = Hf.reshape(sm.m, sm.m)
    Hf_img = np.clip(Hf_img, 0, 1)
    Hf_pil = Image.fromarray((Hf_img * 255).astype(np.uint8), mode="L")

    fig, ax = plt.subplots(figsize=Hf_img.shape[::-1], dpi=1, tight_layout=True)
    ax.imshow(Hf_pil, cmap="gray")
    ax.axis("off")
    fig.savefig(f"{DATA_PATH}/240825/{SAMPLE_NAME}.png", dpi=1)
    plt.show()
