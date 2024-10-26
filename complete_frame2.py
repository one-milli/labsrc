# %%
import cv2
import numpy as np


def load_frames(frame_a_path, frame_b_path):
    """
    フレームAとフレームBを読み込みます。
    """
    frame_a = cv2.imread(frame_a_path)
    frame_b = cv2.imread(frame_b_path)

    if frame_a is None or frame_b is None:
        raise FileNotFoundError("フレームAまたはフレームBの読み込みに失敗しました。")

    return frame_a, frame_b


def compute_optical_flow(frame_a, frame_b):
    """
    フレームAからフレームBへのオプティカルフローを計算します。
    """
    # オプティカルフローの計算（Farneback法）
    flow = cv2.calcOpticalFlowFarneback(
        frame_a,
        frame_b,
        None,
        0.5,  # pyramid scale
        3,  # levels
        11,  # window size
        3,  # iterations
        5,  # poly_n
        1.2,  # poly_sigma
        0,  # flags
    )

    return flow


def warp_frame(frame, flow, scale=0.5):
    """
    フレームをワープします。scaleは動きベクトルのスケール係数です。
    """
    h, w = flow.shape[:2]
    # 動きベクトルをスケーリング
    flow_scaled = flow * scale

    # メッシュグリッドの作成
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow_scaled[..., 0]).astype(np.float32)
    map_y = (grid_y + flow_scaled[..., 1]).astype(np.float32)

    # ワーピング
    warped = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return warped


def estimate_middle_frame(frame_a, frame_b, flow_a_to_b):
    """
    中間フレームCを推定します。
    """
    # フレームAからCへのワープ（スケール0.5）
    warped_a = warp_frame(frame_a, flow_a_to_b, scale=0.5)

    # フレームBからCへのワープ（逆方向の動きベクトル）
    flow_b_to_a = -flow_a_to_b
    warped_b = warp_frame(frame_b, flow_b_to_a, scale=0.5)

    # ブレンディング（単純な平均）
    frame_c = cv2.addWeighted(warped_a, 0.5, warped_b, 0.5, 0)

    return frame_c


# %%
DATA_PATH = "../data"
H_SETTING = "hadamard_FISTA_p-5_lmd-1_m-128"


def matrix_to_tensor(H, m, n):
    return H.reshape(m, m, n, n)


H = np.load(f"{DATA_PATH}/241022/systemMatrix/H_matrix_{H_SETTING}.npy")
H_tensor = matrix_to_tensor(H, 128, 128)

# %%
# フレームの読み込み
frame_a = H_tensor[63, 63, :, :]
frame_b = H_tensor[65, 65, :, :]

# オプティカルフローの計算
flow_a_to_b = compute_optical_flow(frame_a, frame_b)

# 中間フレームCの推定
frame_c = estimate_middle_frame(frame_a, frame_b, flow_a_to_b)

# 結果の表示
cv2.imshow("Frame A", frame_a)
cv2.imshow("Frame B", frame_b)
cv2.imshow("Estimated Frame C", frame_c)

# 結果の保存（オプション）
# cv2.imwrite('frame_c.png', frame_c)

# キー入力待ち
cv2.waitKey(0)
cv2.destroyAllWindows()
