"""Change filename"""

import os

# 変更を行いたいフォルダのパス
DATA_PATH = "../data"
FOLDER_PATH = DATA_PATH + "/hadamard128_input/"

for filename in os.listdir(FOLDER_PATH):
    if filename.startswith("hadamard") and filename.endswith(".png"):
        # ファイル名を分割して新しい名前を作成します。
        parts = filename.split("hadamard")
        new_filename = "hadamard_" + parts[1]

        os.rename(os.path.join(FOLDER_PATH, filename), os.path.join(FOLDER_PATH, new_filename))

print("ファイル名の変更が完了しました。")
