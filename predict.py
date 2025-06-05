import time
import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import copy
import torch
import torch.nn.functional as F


def preprocess_input(image):
    image /= 255.0
    return image


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh

def detect_image(model,image, input_shape,num_classes, count=False, name_classes=None, return_pr=False, device='cpu', mix_type=0):
    if num_classes <= 21:
        colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                       (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
                       (192, 0, 128),
                       (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                       (0, 64, 128),
                       (128, 64, 12)]
    assert isinstance(input_shape,list) #输入图像尺寸应该为列表
    # ---------------------------------------------------------#
    #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    # ---------------------------------------------------------#
    image = cvtColor(image)

    # ---------------------------------------------------#
    #   对输入图像进行一个备份，后面用于绘图
    # ---------------------------------------------------#
    old_img = copy.deepcopy(image)
    orininal_h = np.array(image).shape[0]
    orininal_w = np.array(image).shape[1]
    # ---------------------------------------------------------#
    #   给图像增加灰条，实现不失真的resize
    #   也可以直接resize进行识别
    # ---------------------------------------------------------#
    image_data, nw, nh = resize_image(image, (input_shape[1], input_shape[0]))
    # ---------------------------------------------------------#
    #   添加上batch_size维度
    # ---------------------------------------------------------#
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

    with torch.no_grad():
        images = torch.from_numpy(image_data)
        if device:
            images = images.to(device)

        # ---------------------------------------------------#
        #   图片传入网络进行预测
        # ---------------------------------------------------#
        #pr = model(images)[0]
        result = model(images)
        if isinstance(result, tuple):
            _ , pr = result
            pr = pr.squeeze(0)
        elif isinstance(result, dict):
            pr = result['out'][0]
        else:
            pr = result[0]
        # ---------------------------------------------------#
        #   取出每一个像素点的种类
        # ---------------------------------------------------#
        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
        # --------------------------------------#
        #   将灰条部分截取掉
        # --------------------------------------#
        pr = pr[int((input_shape[0] - nh) // 2): int((input_shape[0] - nh) // 2 + nh), \
             int((input_shape[1] - nw) // 2): int((input_shape[1] - nw) // 2 + nw)]
        # ---------------------------------------------------#
        #   进行图片的resize
        # ---------------------------------------------------#
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
        # ---------------------------------------------------#
        #   取出每一个像素点的种类
        # ---------------------------------------------------#
        pr = pr.argmax(axis=-1)



    # ---------------------------------------------------------#
    #   计数
    # ---------------------------------------------------------#
    if count:
        classes_nums = np.zeros([num_classes])
        total_points_num = orininal_h * orininal_w
        print('-' * 63)
        print("|%25s | %15s | %15s|" % ("Key", "Value", "Ratio"))
        print('-' * 63)
        for i in range(num_classes):
            num = np.sum(pr == i)
            ratio = num / total_points_num * 100
            if num > 0:
                print("|%25s | %15s | %14.2f%%|" % (str(name_classes[i]), str(num), ratio))
                print('-' * 63)
            classes_nums[i] = num
        print("classes_nums:", classes_nums)

    if mix_type == 0:
        # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
        # for c in range(num_classes):
        #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
        #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
        #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
        seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
        # ------------------------------------------------#
        #   将新图片转换成Image的形式
        # ------------------------------------------------#
        image = Image.fromarray(np.uint8(seg_img))
        # ------------------------------------------------#
        #   将新图与原图及进行混合
        # ------------------------------------------------#
        image = Image.blend(old_img, image, 0.7)

    elif mix_type == 1:
        # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
        # for c in range(num_classes):
        #     seg_img[:, :, 0] += ((pr[:, :] == c ) * colors[c][0]).astype('uint8')
        #     seg_img[:, :, 1] += ((pr[:, :] == c ) * colors[c][1]).astype('uint8')
        #     seg_img[:, :, 2] += ((pr[:, :] == c ) * colors[c][2]).astype('uint8')
        seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
        # ------------------------------------------------#
        #   将新图片转换成Image的形式
        # ------------------------------------------------#
        image = Image.fromarray(np.uint8(seg_img))

    elif mix_type == 2:
        seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
        # ------------------------------------------------#
        #   将新图片转换成Image的形式
        # ------------------------------------------------#
        image = Image.fromarray(np.uint8(seg_img))
        # Save npy result
    if return_pr:
        return image, pr

    return image



if __name__ == "__main__":
    mode = "batch_predict"
    model_path = 'ckpt/best_model/best_epoch_weights.pth'
    count = False
    name_classes = ["background", "1", "2"]
    img_size = [192,192]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_origin_path = "./test_imgs"  # 设置测试集图片的文件夹路径
    batch_save_path = "results"  # 设置预测结果保存的路径


    from nets.GEAUNet import self_net
    model = self_net(n_classes=4).eval().to(device)

    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict)
    model.eval()
    if mode == "batch_predict":
        img_names = os.listdir(batch_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(batch_origin_path, img_name)
                image = Image.open(image_path)
                r_image, pr = detect_image(model,image,input_shape=img_size,num_classes=4,count=count,name_classes=name_classes,return_pr=True,device=device)  # 确保 detect_image 函数返回 pr

                if not os.path.exists(batch_save_path):
                    os.makedirs(batch_save_path)
                r_image.save(os.path.join(batch_save_path, img_name))

                npy_save_path = os.path.join(batch_save_path, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                np.save(os.path.join(npy_save_path, "prediction_" + img_name.replace('.jpg', '.npy')), pr)

    else:
        raise AssertionError(
            "Please specify the correct mode: 'predict', 'video', 'fps', 'dir_predict', 'export_onnx', 'predict_onnx', or 'batch_predict'."
        )