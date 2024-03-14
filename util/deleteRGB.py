from PIL import Image

# 画像のパスを指定
image_path = '../../data/231222/random_noise.png'
output_path = '../../data/231222/random_noise_R.png'

# 画像を読み込む
image = Image.open(image_path)
pixels = image.load()

# 画像の各ピクセルを処理
for i in range(image.width):
    for j in range(image.height):
        r, g, b, a = pixels[i, j]
        pixels[i, j] = (r, 0, 0, a)

# 編集した画像を保存
image.save(output_path)
