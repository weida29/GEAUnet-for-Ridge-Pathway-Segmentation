from PIL import Image


def concatenate_images(image_path1, image_path2, output_path, separator_width=20):
    # 打开图像
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)

    # 获取图像的宽度和高度
    width1, height1 = img1.size
    width2, height2 = img2.size

    # 创建一个新图像，宽度为两张图像的宽度之和加上分隔条的宽度，高度为两者中较高的高度
    new_width = width1 + width2 + separator_width
    new_height = max(height1, height2)
    new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))  # 背景为白色

    # 将第一张图像粘贴到新图像上
    new_image.paste(img1, (0, 0))

    # 在两张图片之间粘贴分隔条
    new_image.paste((255, 255, 255), (width1, 0, width1 + separator_width, new_height))

    # 将第二张图像粘贴到新图像上
    new_image.paste(img2, (width1 + separator_width, 0))

    # 保存拼接后的图像
    new_image.save(output_path)


# 示例用法

if __name__ == "__main__":
    image_path1 = 'output_image_original.jpg'  # 第一张图像路径
    image_path2 = './visualize/ground_truth/original_img/scratches.jpg'  # 第二张图像路径
    output_path = 'output_image_original.jpg'  # 输出图像路径

    concatenate_images(image_path1, image_path2, output_path)

    print(f"Images concatenated and saved as {output_path}")