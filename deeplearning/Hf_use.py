import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from Hf import DoubleConv, UNet


# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルのインスタンス化と学習済みの状態辞書のロード
model = UNet(in_channels=3, out_channels=3).to(device)
model.load_state_dict(torch.load(
    '../../data/231222/unet_model_RGB_MSE.pth', map_location=device))
model.eval()  # 推論モードに切り替え

# 画像の変換ルールを定義
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


def image_loader(image_name):
    image = Image.open(image_name).convert("RGB")
    image = transform(image).float()
    image = image.unsqueeze(0)  # バッチ次元の追加
    return image.to(device)


def save_image(tensor, filename):
    image = tensor.cpu().clone()  # テンソルを複製してCPUに移動
    image = image.squeeze(0)      # バッチ次元の削除
    image = transforms.ToPILImage()(image)
    image.save(filename)


# 新しい入力画像のロード
filename = "Cameraman_B"
input_image = image_loader("../../data/231222/" + filename + ".png")

# モデルを通じて画像を変換
with torch.no_grad():  # 勾配計算を行わない
    output_image = model(input_image)

# 結果の画像を保存
output_filepath = "../../data/231222/" + filename + "_RGB_MSE.png"
save_image(output_image, output_filepath)

# 出力画像を表示
plt.imshow(Image.open(output_filepath))
plt.show()
