import os
import math
import re
import random
import numpy as np
import cupy as cp
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt


def calculate_bias(px_cnt, data_path, cap_date):
    black_img1 = Image.open(f"{data_path}/capture_{cap_date}/Black1.png").convert("L")
    black_img2 = Image.open(f"{data_path}/capture_{cap_date}/Black2.png").convert("L")
    black_img3 = Image.open(f"{data_path}/capture_{cap_date}/Black3.png").convert("L")
    black_img4 = Image.open(f"{data_path}/capture_{cap_date}/Black4.png").convert("L")
    black_img5 = Image.open(f"{data_path}/capture_{cap_date}/Black5.png").convert("L")
    black1 = cp.asarray(black_img1) / 255
    black2 = cp.asarray(black_img2) / 255
    black3 = cp.asarray(black_img3) / 255
    black4 = cp.asarray(black_img4) / 255
    black5 = cp.asarray(black_img5) / 255
    average = (black1 + black2 + black3 + black4 + black5) / 5
    average_value = cp.mean(average)
    bias_vector = cp.full(px_cnt, average_value)
    print(f"average_value: {average_value}")
    return bias_vector


def get_use_list(n, p):
    """
    1以上n以下の数からランダムにn*p個(0 < p <= 1)を選び、ソートされたリストを返す関数。

    Parameters:
    n (int): 選択範囲の上限（1以上の整数）。
    p (float): 選択割合（0 < p <= 1）。

    Returns:
    list: ソートされたランダムに選ばれた整数のリスト。
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("nは1以上の整数でなければなりません。")
    if not 0 < p <= 1:
        raise ValueError("pは0より大きく1以下の浮動小数点数でなければなりません。")

    k = math.floor(n * p - 1)
    if k == 0:
        raise ValueError("選択する個数が0になります。")

    SEED = 1
    rng = random.Random(SEED)

    selected = rng.sample(range(2, n + 1), k)
    selected.append(1)
    selected.sort()

    selected = np.array(selected)
    # save as npy file
    # np.save(f"use_list_p-{p}.npy", selected)

    return selected


def images2matrix(folder_path, use_list, thin_out=False):
    """
    画像フォルダ内の画像を読み込み、指定されたインデックスの画像をフラット化して行列に変換する関数。

    Parameters:
    folder_path (str): 画像フォルダのパス。
    use_list (ndarray): 使用する画像のインデックスのリスト。
    thin_out (bool): 画像を間引くかどうか。

    Returns:
    np.ndarray: 画像の行列。
    """
    files = os.listdir(folder_path)
    files.sort(key=lambda f: int(re.search(r"(\d+).png", f).group(1)))

    images = []
    for i in use_list.tolist():
        img = Image.open(os.path.join(folder_path, files[i - 1])).convert("L")
        img_array = cp.asarray(img) / 255
        if thin_out:
            img_array = img_array[::2, ::2].ravel()
        else:
            img_array = img_array.ravel()
        images.append(img_array)

    return np.column_stack(images)


def images_to_matrix(folder_path, convert_gray=True, rand=True, ratio=1.0, resize=False, ressize=255, thin_out=False):
    SEED = 5
    IMG_NAME = "hadamard"
    files = os.listdir(folder_path)
    files.sort(key=lambda f: int(re.search(f"{IMG_NAME}_(\d+).png", f).group(1)))
    if rand:
        random.seed(SEED)
        random.shuffle(files)

    total_files = len(files)
    number_of_files_to_load = int(total_files * ratio)
    selected_files = files[:number_of_files_to_load]
    selected_files.sort(key=lambda f: int(re.search(f"{IMG_NAME}_(\d+).png", f).group(1)))

    images = []
    use_list = []

    # for file in selected_files:
    for i, file in enumerate(selected_files):
        index = int(re.sub(r"\D", "", file))
        use_list.append(index)
        img = Image.open(os.path.join(folder_path, file))
        if i == 0:
            print(f"dtype: {np.array(img).dtype}")
        if convert_gray:
            img = img.convert("L")
        if resize:
            img = img.resize((ressize, ressize), Image.Resampling.BICUBIC)
        if thin_out:
            img_array = np.asarray(img)
            img_array = img_array[::2, ::2]
            img = Image.fromarray(img_array)
        img_array = cp.asarray(img).flatten()
        img_array = img_array / 255
        images.append(img_array)

    return cp.column_stack(images), use_list


# def find_low_pixel_indices(image_path, threshold, apply_noise_reduction=False, blur_radius=2):
#     """
#     画像を読み込み、ノイズ除去（オプション）、フラット化して、画素値がthresholdより小さい画素のインデックスを返す。

#     Parameters:
#     - image_path (str): 画像ファイルのパス。
#     - threshold (int or float): 閾値。
#     - apply_noise_reduction (bool): ノイズ除去を適用するかどうか。
#     - blur_radius (int or float): ガウシアンブラーの半径。

#     Returns:
#     - indices (numpy.ndarray): 条件を満たす画素のインデックス配列。
#     - original_shape (tuple): 画像の元の形状。
#     - image_array (numpy.ndarray): 処理後の画像のNumPy配列。
#     """
#     image = Image.open(image_path).convert("L")

#     # ノイズ除去
#     if apply_noise_reduction:
#         image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

#     image_array = np.array(image)
#     original_shape = image_array.shape

#     flat_image = image_array.flatten()

#     # 閾値より小さい画素のインデックスを取得
#     indices = np.where(flat_image < threshold)[0]

#     return indices, original_shape, image_array


# def create_heatmap(image_array, threshold):
#     """
#     画像データからヒートマップを作成し、閾値より小さい画素を強調表示する。

#     Parameters:
#     - image_array (numpy.ndarray): 画像のNumPy配列。
#     - threshold (int or float): 閾値。
#     """
#     # 閾値より小さい画素をマスク
#     mask = image_array < threshold

#     plt.figure(figsize=(8, 6))
#     plt.imshow(image_array, cmap="gray", interpolation="nearest")
#     plt.colorbar(label="Pixel Intensity")

#     red_overlay = np.zeros_like(image_array)
#     red_overlay[mask] = 255

#     # 赤色のチャネルを追加
#     plt.imshow(red_overlay, cmap="Reds", alpha=0.1)

#     plt.title(f"Heatmap of Pixels Below Threshold ({threshold})")
#     plt.axis("off")
#     plt.show()


# def delete_rows(matrix: np.ndarray, indices: list) -> np.ndarray:
#     """
#     指定したインデックスの行を削除した新しい行列を返します。

#     Parameters:
#     - matrix (np.ndarray): 元の行列
#     - indices (list): 削除する行のインデックスのリスト

#     Returns:
#     - np.ndarray: 行が削除された新しい行列
#     """
#     return np.delete(matrix, indices, axis=0)


# def revive_rows(reduced_matrix: np.ndarray, indices: list, total_rows: int) -> np.ndarray:
#     """
#     削除された行を0ベクトルとして復活させた行列を返します。

#     Parameters:
#     - reduced_matrix (np.ndarray): 行が削除された行列
#     - indices (list): 復活させる行のインデックスのリスト
#     - total_rows (int): 元の行列の総行数

#     Returns:
#     - np.ndarray: 指定した行が0ベクトルとして復活された行列
#     """
#     revived_matrix = np.zeros((total_rows, reduced_matrix.shape[1]), dtype=reduced_matrix.dtype)

#     remaining_indices = [i for i in range(total_rows) if i not in indices]

#     revived_matrix[remaining_indices, :] = reduced_matrix

#     return revived_matrix
