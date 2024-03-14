import os

# 変更を行いたいフォルダのパスを指定します。
folder_path = '../../data/hadamard64_cap_W_sim/'

# フォルダ内のすべてのファイルをループ処理します。
for filename in os.listdir(folder_path):
    if filename.startswith("hadamard") and filename.endswith(".png"):
        # ファイル名を分割して新しい名前を作成します。
        parts = filename.split('hadamard64')
        new_filename = 'hadamard' + parts[1]

        # ファイル名を変更します。
        os.rename(os.path.join(folder_path, filename),
                  os.path.join(folder_path, new_filename))

print("ファイル名の変更が完了しました。")
