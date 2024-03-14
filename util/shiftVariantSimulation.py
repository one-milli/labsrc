import os
from PIL import Image
import numpy as np
import cv2


def adjust_alpha(image, alpha):
    """ Adjust the alpha channel of an RGBA image """
    r, g, b, a = image.split()
    a = a.point(lambda p: int(p * alpha))
    return Image.merge('RGBA', (r, g, b, a))


def transform_image(original_image):
    m = 128
    resize = m*3//4
    origin = m//8
    # Resize the image to 90x90
    resized_image = original_image.resize((resize, resize), Image.ANTIALIAS)

    # Create a mxm transparent canvas
    canvas = Image.new('RGBA', (m, m), (0, 0, 0, 0))
    canvas.paste(resized_image, (origin, origin),
                 resized_image)  # Paste with alpha

    # Convert to numpy array for OpenCV processing
    canvas_np = np.array(canvas)

    # Points in the source image
    rows, cols, _ = canvas_np.shape
    src_points = np.float32(
        [[origin, origin], [m-origin, origin], [origin, m-origin], [m-origin, m-origin]])

    # Points in the destination image
    top_width = resize * 9 / 10
    bottom_width = resize
    dst_points = np.float32([[cols / 2 - top_width / 2, origin],
                             [cols / 2 + top_width / 2, origin],
                             [origin, m-origin],
                             [m-origin, m-origin]])

    # Transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the transformation
    warped_image = cv2.warpPerspective(
        canvas_np, matrix, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # Convert back to PIL image with alpha
    trapezoid_image = Image.fromarray(warped_image, 'RGBA')

    # Adjust alpha to 50%
    trapezoid_image = adjust_alpha(trapezoid_image, 0.5)

    # Create a new transparent canvas for stacking trapezoid images
    stacked_canvas = Image.new('RGBA', (m, m), (0, 0, 0, 0))

    # Stack the trapezoid images with vertical offset
    offset = 3
    for i in range(3):
        if i == 1:  # Second image (90% scale)
            scaled_image = trapezoid_image.resize(
                (m, int(m * 0.9)), Image.ANTIALIAS)
        elif i == 2:  # Third image (80% scale)
            scaled_image = trapezoid_image.resize(
                (m, int(m * 0.8)), Image.ANTIALIAS)
        else:  # First image (no scaling)
            scaled_image = trapezoid_image

        stacked_canvas.paste(
            scaled_image, (0, offset * i), scaled_image)

    # Create a new canvas with black background
    final_canvas = Image.new('RGB', (m, m), (0, 0, 0))

    # Paste the stacked image onto the black background
    final_canvas.paste(stacked_canvas)

    return final_canvas


def process_images(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Load the image with alpha channel (transparency)
            original_image = Image.open(input_path).convert("RGBA")

            # Apply the transformation and stacking
            final_image = transform_image(original_image)

            # Save the final image
            final_image.save(output_path, format="PNG")


# Example usage
input_folder = '../../data/Hadamard64_input/'
output_folder = '../../data/hadamard64_cap_W_sim/'
process_images(input_folder, output_folder)
