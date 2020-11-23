import cv2
from torch_py.FaceRec import Recognition
from torch_py.MTCNN.detector import FaceDetector
from torch_py.Utils import plot_image
import torch
from torch_py.MobileNetV2 import MobileNetV2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


# data_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),  # 转化为张量
#     transforms.Normalize([0], [1]),  # 归一化
# ])  # T: torchvision.transforms
# # load image
# img = Image.open("mask_3.jpg")
# plt.imshow(img)
# # [N, C, H, W]
# img = data_transform(img)
# # expand batch dimension
# img = torch.unsqueeze(img, dim=0)

# # read class_indict
# class_indict = ["mask", "no_mask"]

# # create model
# model = MobileNetV2(classes=2)
# # load model weights
# model_weight_path = "results/temp.pth"
# model.load_state_dict(torch.load(model_weight_path))
# model.eval()
# with torch.no_grad():
#     # predict class
#     output = torch.squeeze(model(img))
#     predict = torch.softmax(output, dim=0)
#     predict_cla = torch.argmax(predict).numpy()

# print(class_indict[predict_cla], predict[predict_cla].numpy())
# plt.show()


# -------------------------- 请加载您最满意的模型 ---------------------------
# 加载模型(请加载你认为的最佳模型)
# 加载模型,加载请注意 model_path 是相对路径, 与当前文件同级。
# 如果你的模型是在 results 文件夹下的 dnn.h5 模型，则 model_path = 'results/temp.pth'
model_path = 'results/temp.pth'
# ---------------------------------------------------------------------------


def predict(img):
    """
    加载模型和模型预测
    :param img: cv2.imread 图像
    :return: 预测的图片中的总人数、其中佩戴口罩的人数
    """
    # -------------------------- 实现模型预测部分的代码 ---------------------------
    # 将 cv2.imread 图像转化为 PIL.Image 图像，用来兼容测试输入的 cv2 读取的图像（勿删！！！）
    # cv2.imread 读取图像的类型是 numpy.ndarray
    # PIL.Image.open 读取图像的类型是 PIL.JpegImagePlugin.JpegImageFile
    if isinstance(img, np.ndarray):
        # 转化为 PIL.JpegImagePlugin.JpegImageFile 类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    recognize = Recognition(model_path)
    img, all_num, mask_num = recognize.mask_recognize(img)
    # -------------------------------------------------------------------------
    return all_num, mask_num


# 输入图片路径和名称
img = cv2.imread("./test.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = Image.open("./test1.jpg")
all_num, mask_num = predict(img)
# 打印预测该张图片中总人数以及戴口罩的人数
print(all_num, mask_num)

img = Image.open("test1.jpg")
detector = FaceDetector()
recognize = Recognition(model_path='results/temp.pth')
draw, all_num, mask_nums = recognize.mask_recognize(img)
plt.imshow(draw)
plt.show()
print("all_num:", all_num, "mask_num", mask_nums)
