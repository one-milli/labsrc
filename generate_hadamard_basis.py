"""アダマール基底画像の生成"""

import os
import numpy as np
from PIL import Image


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


def main():
    """
    メイン処理関数
    """
    m = 128  # Hadamard画像のサイズを制御
    b = 7  # <m-1> は <b> ビットで表現可能

    n = 1

    output_dir = f"../../OneDrive - m.titech.ac.jp/Lab/data/hadamard{m}_input"
    os.makedirs(output_dir, exist_ok=True)

    for vidx in range(m):
        for uidx in range(m):
            F = get_basis(uidx, vidx, m, b)

            # Fを二値化: F > 0 なら255、そうでなければ0
            F_bin = (F > 0).astype(np.uint8) * 255

            print(n)

            filename = f"hadamard_{n}.png"
            filepath = os.path.join(output_dir, filename)

            img = Image.fromarray(F_bin, mode="L")
            img.save(filepath)

            n += 1


if __name__ == "__main__":
    main()
