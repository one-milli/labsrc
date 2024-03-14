from PIL import Image
import os


def resize_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            image_path = os.path.join(input_folder, filename)
            with Image.open(image_path) as img:
                if img.size == (64, 64):
                    img_resized = img.resize((128, 128))
                    new_filename = f"{os.path.splitext(filename)[0]}.png"
                    img_resized.save(os.path.join(output_folder, new_filename))


# 入力フォルダのパス
input_folder = '../../data/Hadamard64_input/'

# 出力フォルダのパス
output_folder = '../../data/Hadamard64_input128_W/'

resize_images(input_folder, output_folder)
