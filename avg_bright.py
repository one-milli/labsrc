from PIL import Image
import numpy as np


def calculate_average_luminance(image_path):
    """
    画像の平均輝度を計算します。

    Args:
        image_path (str): 画像ファイルのパス。

    Returns:
        float: 画像の平均輝度。
    """
    with Image.open(image_path) as img:
        # グレースケールに変換して輝度を取得
        grayscale = img.convert("L")
        np_pixels = np.array(grayscale)
        average = np_pixels.mean()
    return average


def compute_scaling_factor(reference_avg, target_avg):
    """
    基準画像とターゲット画像の平均輝度に基づいてスケーリング係数を計算します。

    Args:
        reference_avg (float): 基準画像の平均輝度。
        target_avg (float): ターゲット画像の平均輝度。

    Returns:
        float: スケーリング係数。
    """
    if target_avg == 0:
        raise ValueError("ターゲット画像の平均輝度が0です。スケーリングできません。")
    return reference_avg / target_avg


def adjust_brightness(image_path, scaling_factor, output_path):
    """
    画像の輝度を調整し、保存します。

    Args:
        image_path (str): 調整したい画像のパス。
        scaling_factor (float): 輝度調整に使用するスケーリング係数。
        output_path (str): 調整後の画像を保存するパス。
    """
    with Image.open(image_path) as img:
        # RGB各チャネルにスケーリングを適用
        np_pixels = np.array(img).astype(np.float32)
        np_adjusted = np_pixels * scaling_factor
        # ピクセル値を0-255にクリップし、整数に変換
        np_adjusted = np.clip(np_adjusted, 0, 255).astype(np.uint8)
        adjusted_img = Image.fromarray(np_adjusted)
        adjusted_img.save(output_path)
        print(f"調整後の画像を保存しました: {output_path}")


def main():
    # 基準画像と画像Aのパスを指定
    reference_image_path = "reference.jpg"  # 基準となる画像
    target_image_path = "imageA.jpg"  # 輝度を調整したい画像A

    # 平均輝度を計算
    reference_avg = calculate_average_luminance(reference_image_path)
    target_avg = calculate_average_luminance(target_image_path)
    print(f"基準画像の平均輝度: {reference_avg}")
    print(f"画像Aの平均輝度: {target_avg}")

    # スケーリング係数を計算
    scaling_factor = compute_scaling_factor(reference_avg, target_avg)
    print(f"スケーリング係数: {scaling_factor}")

    # 画像Aの輝度を調整して保存
    adjusted_imageA_path = "imageA_adjusted.jpg"
    adjust_brightness(target_image_path, scaling_factor, adjusted_imageA_path)

    # 他の画像にも同じ係数を適用する例
    other_image_path = "other_image.jpg"
    adjusted_other_image_path = "other_image_adjusted.jpg"
    adjust_brightness(other_image_path, scaling_factor, adjusted_other_image_path)


if __name__ == "__main__":
    main()
