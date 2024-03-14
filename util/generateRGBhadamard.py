from PIL import Image
import numpy as np
import os


def create_colored_channels_from_binary(folder_path, output_size=(128, 128)):
    # Create directories for each channel's images
    folder = os.path.join(folder_path, 'Hadamard64_input128')
    os.makedirs(folder, exist_ok=True)

    # Process each image in the folder
    for i in range(1, 4097):  # Assuming numbering from 1 to 4096
        filename = f'hadamard{i}.png'
        image_path = os.path.join(folder_path, filename)
        try:
            # Open the image and resize it
            with Image.open(image_path) as img:
                # Resize the image
                img_resized = img.resize(output_size)
                # Convert to grayscale in case it's not a binary image
                img_gray = img_resized.convert('L')
                # Create a binary mask where the white pixels (255) will be True
                mask = np.array(img_gray) == 255

                # Initialize the RGB channels with zeros (black)
                r_channel = np.zeros(
                    (output_size[1], output_size[0], 3), dtype=np.uint8)
                g_channel = np.zeros_like(r_channel)
                b_channel = np.zeros_like(r_channel)

                r_channel[mask, 0] = 255
                g_channel[mask, 1] = 255
                b_channel[mask, 2] = 255

                # Convert the channels back to image
                r_image = Image.fromarray(r_channel, 'RGB')
                g_image = Image.fromarray(g_channel, 'RGB')
                b_image = Image.fromarray(b_channel, 'RGB')

                # Save the images
                filename = f'hadamard{i}.png'
                r_image.save(os.path.join(folder, filename))
                # filename = f'hadamard{i+4096}.png'
                # g_image.save(os.path.join(folder, filename))
                # filename = f'hadamard{i+8192}.png'
                # b_image.save(os.path.join(folder, filename))
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("Processing complete.")


# Example usage
# Replace 'path_to_your_folder' with the path to the folder containing your binary images
create_colored_channels_from_binary('../data/Hadamard64_input/')
