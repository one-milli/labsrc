import os
from PIL import Image


def resize_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for i in range(1, 4097):  # 1から4096まで
        file_name = f'hadamard64_{i}.png'
        file_path = os.path.join(input_folder, file_name)
        if i % 1024 == 0:
            print(i)

        # ファイルが存在する場合のみ処理を実行
        if os.path.isfile(file_path):
            with Image.open(file_path) as img:
                # トリミングする位置とサイズを指定
                left = 400
                top = 460
                right = 860
                bottom = 920

                # トリミングとリサイズ
                img = img.crop((left, top, right, bottom))
                img = img.resize((128, 128))

                # 画像を出力フォルダに保存
                new_file_name = f'hadamard64_{i+8192}.png'
                img.save(os.path.join(output_folder, new_file_name))


resize_images('../data/hadamard_cap_B_230516/',
              '../data/hadamard_cap_230516_resize128/')
