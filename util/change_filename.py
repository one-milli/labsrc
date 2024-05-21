"""Change filename"""

import os
import localConfig

# 変更を行いたいフォルダのパス
FOLDER_PATH = localConfig.DATA_PATH + '/hadamard64_cap_R_230516_128/'

for filename in os.listdir(FOLDER_PATH):
    if filename.startswith("hadamard") and filename.endswith(".png"):
        # ファイル名を分割して新しい名前を作成します。
        parts = filename.split('hadamard64')
        new_filename = 'hadamard' + parts[1]

        os.rename(os.path.join(FOLDER_PATH, filename),
                  os.path.join(FOLDER_PATH, new_filename))

print("ファイル名の変更が完了しました。")
