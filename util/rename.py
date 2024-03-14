import os

# 対象のフォルダへのパス
folder_path = '../data/Hadamard64_input256_B/'

# フォルダ内のすべてのファイルを取得
files = os.listdir(folder_path)

# 各ファイルに対して名前を変更
for i, file_name in enumerate(files):
    # 新しいファイル名を生成 (例: new_name_1.jpg)
    # new_name = f'hadamard{i+8192}{os.path.splitext(file_name)[1]}'
    new_name = f'hadamard{i}{os.path.splitext(file_name)[1]}'

    # 元のファイルパスと新しいファイルパス
    old_file = os.path.join(folder_path, file_name)
    new_file = os.path.join(folder_path, new_name)

    # ファイル名を変更
    os.rename(old_file, new_file)

print("ファイル名の変更が完了しました。")
