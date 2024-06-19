from PIL import Image, ImageOps
import os

def augment_image(image_path, output_dir):
    image = Image.open(image_path)

    base_name = os.path.basename(image_path).split('.')[0]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image.save(os.path.join(output_dir, f"{base_name}_original.png"))

    flipped_lr_image = ImageOps.mirror(image)
    flipped_lr_image.save(os.path.join(output_dir, f"{base_name}_flipped_lr.png"))

    flipped_tb_image = ImageOps.flip(image)
    flipped_tb_image.save(os.path.join(output_dir, f"{base_name}_flipped_tb.png"))

    for angle in [90, 180, 270]:
        rotated_image = image.rotate(angle, expand=True)

        background = Image.new('RGB', rotated_image.size, (255, 255, 255))

        background.paste(rotated_image, (0, 0), rotated_image.convert('RGBA'))

        background.save(os.path.join(output_dir, f"{base_name}_rotated_{angle}.png"))

def augment_images_in_folder(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            augment_image(file_path, output_folder)

input_folder = 'Generated'
output_folder = 'augmented_base'

augment_images_in_folder(input_folder, output_folder)
