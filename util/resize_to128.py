from PIL import Image
import sys
import os

def resize_square_png(input_path, output_path, target_size=(128, 128)):
    """
    正方形のPNG画像を指定されたサイズにリサイズします。
    
    :param input_path: 入力PNG画像のパス
    :param output_path: リサイズ後の画像を保存するパス
    :param target_size: ターゲットのサイズ (幅, 高さ)
    """
    try:
        with Image.open(input_path) as img:
            # 画像がPNG形式か確認
            if img.format != 'PNG':
                print(f"エラー: {input_path} はPNG形式ではありません。")
                return
            
            width, height = img.size
            print(f"元の画像サイズ: {width}x{height}px")

            # 正方形かどうかを確認
            if width != height:
                print("エラー: 画像が正方形ではありません。")
                return
            
            # サイズがターゲットより大きいか確認
            if width <= target_size[0] and height <= target_size[1]:
                print(f"エラー: 画像のサイズが{target_size[0]}x{target_size[1]}ピクセル以下です。リサイズは必要ありません。")
                return
            
            # リサイズを実行（バイキュービック補間）
            resized_img = img.resize(target_size, Image.BICUBIC)
            resized_img.save(output_path, format='PNG')
            print(f"画像を{target_size[0]}x{target_size[1]}ピクセルにリサイズし、{output_path}に保存しました。")
    
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {input_path}")
    except Exception as e:
        print(f"予期しないエラーが発生しました: {e}")

def main():
    """
    コマンドライン引数を解析し、リサイズ関数を呼び出します。
    使用法:
        python resize_png.py 入力画像.png 出力画像.png
    """
    if len(sys.argv) != 3:
        print("使用法: python resize_png.py 入力画像.png 出力画像.png")
        sys.exit(1)
    
    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]

    # 入力ファイルの存在を確認
    if not os.path.isfile(input_image_path):
        print(f"エラー: 入力ファイルが存在しません - {input_image_path}")
        sys.exit(1)
    
    resize_square_png(input_image_path, output_image_path)

if __name__ == "__main__":
    main()
