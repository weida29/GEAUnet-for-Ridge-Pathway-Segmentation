import numpy as np
from PIL import Image
import os


def png_to_npy(png_path, npy_path):
    # 打开 PNG 文件
    img = Image.open(png_path)

    # 将图像转换为 NumPy 数组
    label_array = np.array(img)

    # 保存为 NPY 文件
    np.save(npy_path, label_array)


def convert_directory(png_directory, npy_directory):
    # 确保目标目录存在
    if not os.path.exists(npy_directory):
        os.makedirs(npy_directory)

    # 遍历目录中的所有文件
    for filename in os.listdir(png_directory):
        if filename.endswith(".png"):
            # 构建完整的文件路径
            png_path = os.path.join(png_directory, filename)
            npy_filename = filename[:-4] + ".npy"
            npy_path = os.path.join(npy_directory, npy_filename)

            # 转换文件
            png_to_npy(png_path, npy_path)
            print(f"Converted: {png_path} -> {npy_path}")


# 示例用法
png_directory_path = '../strawberry_dataset/test/gray_gt_img/'
npy_directory_path = '../strawberry_dataset/test/test_gt/'
convert_directory(png_directory_path, npy_directory_path)
