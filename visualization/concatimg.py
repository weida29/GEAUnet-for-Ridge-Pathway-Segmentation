import cv2
import numpy as np
from PIL import Image
import os


# 用于垂直拼接图像，并在拼接处加入分割线
def concatenate_images_vertically(images, separator_height=5, separator_color=(255,255,255)):
    """
    垂直拼接一系列图像，且在每张图像之间增加细条分隔线。

    :param images: 要拼接的图像列表，每个图像为PIL.Image或numpy数组
    :param separator_height: 分隔线的高度（像素），默认为5像素
    :param separator_color: 分隔线的颜色，默认为黑色
    :return: 拼接后的图像，类型为PIL.Image
    """

    # 确保所有图像的宽度一致，以避免拼接时尺寸不匹配
    widths, heights = zip(*(img.size for img in images))
    total_height = sum(heights) + (len(images) - 1) * separator_height
    max_width = max(widths)

    # 创建拼接后的图像，背景为灰色
    concatenated_image = Image.new('RGB', (max_width, total_height), (128, 128, 128))

    y_offset = 0
    for i, img in enumerate(images):
        concatenated_image.paste(img, (0, y_offset))
        y_offset += img.size[1]

        # 在每两张图像之间添加分隔线
        if i < len(images) - 1:
            separator = Image.new('RGB', (max_width, separator_height), separator_color)
            concatenated_image.paste(separator, (0, y_offset))
            y_offset += separator_height

    return concatenated_image


if __name__ == "__main__":
    # 示例代码：从文件夹加载图像并进行拼接
    batch_origin_path = "./visualize/output_img"  # 设置测试集图片的文件夹路径
    batch_save_path = "./visualize/mask_concat"  # 设置拼接图像保存的路径
    save_img_name = 'caege_aspp'

    img_names = os.listdir(batch_origin_path)
    images = []

    for img_name in img_names:
        if img_name.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path = os.path.join(batch_origin_path, img_name)
            image = Image.open(image_path)
            images.append(image)

    if images:
        # 垂直拼接图像，添加5像素高度的分隔线
        concatenated_image = concatenate_images_vertically(images, separator_height=5)

        # 保存拼接后的图像
        if not os.path.exists(batch_save_path):
            os.makedirs(batch_save_path)
        concatenated_image.save(os.path.join(batch_save_path, f'{save_img_name}.jpg'))

        print("拼接图像已保存到:", os.path.join(batch_save_path, f'{save_img_name}.jpg'))