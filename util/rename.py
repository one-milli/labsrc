import os

DATA_PATH = '../../../OneDrive - m.titech.ac.jp/Lab/data'
folder_path = f"{DATA_PATH}/hadamard64_cap_B_230516_128/"

files = os.listdir(folder_path)

for i, file_name in enumerate(files):
    new_name = f'hadamard_{i+1}{os.path.splitext(file_name)[1]}'

    old_file = os.path.join(folder_path, file_name)
    new_file = os.path.join(folder_path, new_name)

    os.rename(old_file, new_file)

print("ファイル名の変更が完了しました。")
