import os
from PIL import Image

DATA_PATH = '../../OneDrive - m.titech.ac.jp/Lab/data'


def resize_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for i in range(1, 4097):
        file_name = f'hadamard64_{i}.png'
        file_path = os.path.join(input_folder, file_name)
        if i % 1024 == 0:
            print(i)

        if os.path.isfile(file_path):
            with Image.open(file_path) as img:
                left = 400
                top = 460
                right = 860
                bottom = 920

                img = img.crop((left, top, right, bottom))
                img = img.resize((128, 128))

                new_file_name = f'hadamard_{i}.png'
                img.save(os.path.join(output_folder, new_file_name))


if __name__ == '__main__':
    INPUT = f"{DATA_PATH}/hadamard_cap_B_230516/"
    OUTPUT = f"{DATA_PATH}/hadamard64_cap_B_230516_128"
    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)
    resize_images(input_folder=INPUT, output_folder=OUTPUT)
