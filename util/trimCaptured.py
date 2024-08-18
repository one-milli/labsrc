import os
from PIL import Image

DATA_PATH = "../../OneDrive - m.titech.ac.jp/Lab/data"


def resize_images_all(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for i in range(1, 4097):
        file_name = f"hadamard64_{i}.png"
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

                new_file_name = f"hadamard_{i}.png"
                img.save(os.path.join(output_folder, new_file_name))


def crop_and_resize_single_image(image_path, output_path, left, top, right, bottom, new_size=(256, 256)):
    """
    画像をトリミングしてリサイズし、指定したパスに保存する関数

    Args:
        image_path (str): 処理対象の画像ファイルのパス
        output_path (str): 出力画像ファイルのパス
        left (int): トリミング領域の左座標
        top (int): トリミング領域の上座標
        right (int): トリミング領域の右座標
        bottom (int): トリミング領域の下座標
        new_size (tuple): リサイズ後の画像サイズ (幅, 高さ)
    """
    with Image.open(image_path) as img:
        img = img.crop((left, top, right, bottom))
        img = img.resize(new_size)
        img.save(output_path)


if __name__ == "__main__":
    INPUT = f"{DATA_PATH}/capture_240814/"
    OUTPUT = f"{DATA_PATH}/capture_240814/"
    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)

    # Trim and resize all images
    # resize_images_all(input_folder=INPUT, output_folder=OUTPUT)

    # Trim and resize a single image
    single_image_path = os.path.join(INPUT, "Woman_row.png")
    single_output_path = os.path.join(OUTPUT, "Woman.png")
    crop_and_resize_single_image(single_image_path, single_output_path, 370, 350, 870, 850)
