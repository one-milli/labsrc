"""アダマール基底画像の生成（並列処理版）"""

import os
import multiprocessing
import numpy as np
from PIL import Image
from tqdm import tqdm


def is_bit(index, u):
    """
    指定されたインデックスのビットがセットされているかを判定します。

    Parameters:
        index (int): ビットの位置（0から始まる）
        u (int): 判定対象の整数値

    Returns:
        int: ビットがセットされていれば1、そうでなければ0。
    """
    return 1 if (u & (1 << index)) != 0 else 0


def get_g(i, u, b):
    """
    指定されたビット位置のビット値の合計を取得します。

    Parameters:
        i (int): 現在のビット位置
        u (int): 入力値
        b (int): ビット数

    Returns:
        int: is_bit(b - i, u) + is_bit(b - i - 1, u) の結果
    """
    return is_bit(b - i, u) + is_bit(b - i - 1, u)


def get_q(x, y, u, v, b):
    """
    qの計算を行います。

    Parameters:
        x (int): x座標に対応する値
        y (int): y座標に対応する値
        u (int): uインデックス
        v (int): vインデックス
        b (int): ビット数

    Returns:
        int: 計算結果のq
    """
    q = 0
    for idx in range(b):
        q += is_bit(idx, x) * get_g(idx, u, b) + is_bit(idx, y) * get_g(idx, v, b)
    return q


def get_basis(u, v, m, b):
    """
    F行列（基底行列）を生成します。

    Parameters:
        u (int): uインデックス
        v (int): vインデックス
        m (int): 画像のサイズ
        b (int): ビット数

    Returns:
        numpy.ndarray: 計算されたF行列
    """
    F = np.zeros((m, m), dtype=float)
    for yidx in range(m):
        for xidx in range(m):
            q = get_q(xidx, yidx, u, v, b)
            F[yidx, xidx] = (-1) ** q
    F = F / m
    return F


def generate_and_save_image(args):
    """
    画像を生成し、保存します。

    Parameters:
        args (tuple): (u, v, m, b, output_dir)

    Returns:
        int: 生成した画像の番号
    """
    u, v, m, b, output_dir = args
    F = get_basis(u, v, m, b)

    # Fを二値化: F > 0 なら255、そうでなければ0
    F_bin = (F > 0).astype(np.uint8) * 255

    # nを計算（1から始まる連番）
    n = u * m + v + 1

    filename = f"hadamard_{n}.png"
    filepath = os.path.join(output_dir, filename)

    # 画像を保存
    img = Image.fromarray(F_bin, mode="L")
    img.save(filepath)

    return n


def main():
    """
    メイン処理関数（並列処理版）
    """
    m = 128  # Hadamard画像のサイズを制御
    b = 7  # <m-1> は <b> ビットで表現可能

    output_dir = f"../../OneDrive - m.titech.ac.jp/Lab/data/hadamard{m}_input"
    os.makedirs(output_dir, exist_ok=True)

    # 全ての(u, v)ペアを生成
    tasks = [(u, v, m, b, output_dir) for u in range(m) for v in range(m)]

    total_tasks = len(tasks)
    print(f"Total images to generate and save: {total_tasks}")

    # プロセス数をCPUコア数に設定
    num_processes = multiprocessing.cpu_count()
    print(f"Using {num_processes} parallel processes.")

    with multiprocessing.Pool(processes=num_processes) as pool:
        # tqdmを使用して進行状況を表示
        results = list(tqdm(pool.imap(generate_and_save_image, tasks), total=total_tasks))

    print(f"Generated and saved {total_tasks} images.")


if __name__ == "__main__":
    main()
