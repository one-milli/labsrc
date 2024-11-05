import numpy as np
import matplotlib.pyplot as plt


def create_histogram(npy_file_path):
    """
    指定された .npy ファイルを読み込み、データを範囲ごとに分類し、
    ヒストグラムを作成して表示します。

    Parameters:
    - npy_file_path: str, .npy ファイルのパス
    """
    # 1. .npy ファイルの読み込み
    try:
        data = np.load(npy_file_path)
    except Exception as e:
        print(f"Error loading .npy file: {e}")
        return

    # データが1次元配列であることを確認
    if data.ndim != 1:
        print("Error: データは1次元の配列である必要があります。")
        return

    # 2. ビン（範囲）の定義
    # 最小値と最大値を取得
    min_val = np.min(data)
    max_val = np.max(data)

    # 最初のビンは -inf から 0 まで（<=0）
    # その後は 0 より大きく 0.001 毎のビン
    bin_width = 0.001  # ビンの幅
    # 必要なビン数を計算
    num_bins = int(np.ceil(max_val / bin_width)) if max_val > 0 else 0
    # ビンのエッジを作成
    bin_edges = np.concatenate(([-np.inf, 0], np.arange(bin_width, (num_bins + 1) * bin_width, bin_width)))

    # 3. ヒストグラムの計算
    hist, edges = np.histogram(data, bins=bin_edges)

    # 4. ビンのラベル作成
    labels = ["<=0"]
    for i in range(1, len(edges) - 1):
        lower = edges[i]
        upper = edges[i + 1]
        labels.append(f"{lower:.3f}~{upper:.3f}")

    # 5. ヒストグラムのプロット
    plt.figure(figsize=(12, 6))
    plt.bar(labels, hist, width=0.8, color="skyblue", edgecolor="black")
    plt.xlabel("値の範囲")
    plt.ylabel("頻度")
    plt.title(".npy データのヒストグラム")
    plt.xticks(rotation=90)  # ラベルを縦に表示
    plt.tight_layout()  # レイアウトを調整してラベルが重ならないようにする
    plt.show()


if __name__ == "__main__":
    m = 128
    LAMBDA = 100
    RATIO = 0.05
    DATA_PATH = "../data"
    DIRECTORY = DATA_PATH + "/241105"
    SETTING = f"p-{int(100*RATIO)}_lmd-{LAMBDA}_m-{m}"
    npy_file = f"{DIRECTORY}/systemMatrix/H_matrix_{SETTING}.npy"
    create_histogram(npy_file)
