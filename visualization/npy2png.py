import os
import numpy as np
from PIL import Image


def convert_npy_to_mask(npy_file, save_path, colors):
    # 加载 .npy 文件
    pr = np.load(npy_file)

    # 获取图像的高和宽
    orininal_h, orininal_w = pr.shape

    # 根据预测结果将整数转换为颜色
    seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])

    # 将生成的掩码图像进行保存
    mask_image = Image.fromarray(seg_img)
    mask_image.save(save_path)


if __name__ == "__main__":
    npy_dir = "./visualize/output_img/npy"
    mask_save_dir = "./visualize/model_mask_img/caege"  # 掩码保存路径

    if not os.path.exists(mask_save_dir):
        os.makedirs(mask_save_dir)

    # 定义颜色映射，与原始代码中一致
    colors = [
        (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
        (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
        (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
        (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
        (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
        (0, 64, 128), (128, 64, 12)
    ]

    npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
    for npy_file in npy_files:
        npy_path = os.path.join(npy_dir, npy_file)
        mask_save_path = os.path.join(mask_save_dir, npy_file.replace('.npy', '.png'))

        convert_npy_to_mask(npy_path, mask_save_path, colors)