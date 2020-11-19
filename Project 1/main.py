# %%
import copy

import cv2  # oepnCV
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm  # 进度条

from torch_py.FaceRec import Recognition
from torch_py.MTCNN.detector import FaceDetector
from torch_py.Utils import plot_image

# import warnings
# # 忽视警告
# warnings.filterwarnings('ignore')

# %%
# 1.加载数据并进行数据处理
# 形成数据集，测试集DataLoader


def processing_data(data_path, height=224, width=224, batch_size=32,
                    test_split=0.1):
    """
    数据处理部分
    :param data_path: 数据路径
    :param height:高度
    :param width: 宽度
    :param batch_size: 每次读取图片的数量
    :param test_split: 测试集划分比例
    :return:
    """
    transforms = T.Compose([
        T.Resize((height, width)),
        T.RandomHorizontalFlip(0.1),  # 进行随机水平翻转
        T.RandomVerticalFlip(0.1),  # 进行随机竖直翻转
        T.ToTensor(),  # 转化为张量
        T.Normalize([0], [1]),  # 归一化
    ])  # T: torchvision.transforms

    dataset = ImageFolder(data_path, transform=transforms)
    # 划分数据集
    train_size = int((1-test_split)*len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size])
    # 创建一个 DataLoader 对象
    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)

    return train_data_loader, valid_data_loader


data_path = "./datasets/5f680a696ec9b83bb0037081-momodel/data/image"

train_data_loader, valid_data_loader = processing_data(
    data_path=data_path, height=160, width=160, batch_size=32)


# %%
# 2.如果有预训练模型，则加载预训练模型；如果没有则不需要加载
pnet_path = "./torch_py/MTCNN/weights/pnet.npy"
rnet_path = "./torch_py/MTCNN/weights/rnet.npy"
onet_path = "./torch_py/MTCNN/weights/onet.npy"

# %%
# 3.创建模型和训练模型，训练模型时尽量将模型保存在 results 文件夹


class MobileNetV1(nn.Module):
    def __init__(self, classes=2):
        # classes: 分类数
        super(MobileNetV1, self).__init__()
        self.mobilebone = nn.Sequential(
            self._conv_bn(3, 32, 2),
            self._conv_dw(32, 64, 1),
            # self._conv_dw(64, 128, 2),
            # self._conv_dw(128, 128, 1),
            # self._conv_dw(128, 256, 2),
            # self._conv_dw(256, 256, 1),
            # self._conv_dw(256, 512, 2),
            # self._top_conv(512, 512, 5),
            # self._conv_dw(512, 1024, 2),
            # self._conv_dw(1024, 1024, 1),
        )
        # self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** .5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.mobilebone(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return out

    def _top_conv(self, in_channel, out_channel, blocks):
        layers = []
        for i in range(blocks):
            layers.append(self._conv_dw(in_channel, out_channel, 1))
        return nn.Sequential(*layers)

    def _conv_bn(self, in_channel, out_channel, stride):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3,
                      stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def _conv_dw(self, in_channel, out_channel, stride):
        return nn.Sequential(
            # nn.Conv2d(in_channel, in_channel, 3, stride, 1, groups=in_channel, bias=False),
            # nn.BatchNorm2d(in_channel),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=False),
        )

# %%


device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model = MobileNetV1(classes=2).to(device)

# 优化器
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# 学习率下降的方式，acc三次不下降就下降学习率继续训练，衰减学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 'max',
                                                 factor=0.5,
                                                 patience=2)
# 损失函数
criterion = nn.CrossEntropyLoss()

print('加载完成...')


# %%
# 训练网络
epochs = 20
best_loss = 1e9
best_model_weights = copy.deepcopy(model.state_dict())
loss_list = []  # 存储损失函数值
for epoch in range(epochs):
    model.train()

    for batch_idx, data in tqdm(enumerate(train_data_loader, 1)):
        # get the inputs, data is a list of [inputs, labels]
        x, y = data
        x = x.to(device)
        y = y.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # print(pred_y.shape)
        # print(y.shape)
        # forward + backward + optimize
        pred_y = model(x)
        loss = criterion(pred_y, y)
        loss.backward()
        optimizer.step()
        # 记录损失
        loss_list.append(loss)

        if loss < best_loss:
            best_model_weights = copy.deepcopy(model.state_dict())
            best_loss = loss

    print('step:' + str(epoch + 1) + '/' +
          str(epochs) + ' || Total Loss: %.4f' % (loss.item()))

SAVE_PATH = 'results/temp.pth'
torch.save(model.state_dict(), SAVE_PATH)
print('Finish Training.')

# loss
plt.plot(loss_list, label="loss")
plt.legend()
plt.show()


# %%
img = Image.open("test1.jpg")
detector = FaceDetector()
recognize = Recognition(model_path='results/temp.pth')
draw, all_num, mask_nums = recognize.mask_recognize(img)
plt.imshow(draw)
plt.show()
print("all_num:", all_num, "mask_num", mask_nums)

# %%
# 4.评估模型，将自己认为最佳模型保存在 result 文件夹，其余模型备份在项目中其它文件夹，方便您加快测试通过。


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
img = cv2.imread("test1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
all_num, mask_num = predict(img)
# 打印预测该张图片中总人数以及戴口罩的人数
print(all_num, mask_num)

# %%
