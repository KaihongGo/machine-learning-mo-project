# %%
from torch import nn
import copy
import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, dataset
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm  # 进度条

from torch_py.FaceRec import Recognition
from torch_py.MTCNN.detector import FaceDetector
from torch_py.Utils import plot_image
from torch_py.MobileNetV2 import MobileNetV2
# import warnings
# # 忽视警告
# warnings.filterwarnings('ignore')

# %%
# 1.加载数据并进行数据处理
# 形成数据集，测试集DataLoader


def processing_data(data_path, height=224, width=224, batch_size=20,
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


# %%


device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model = MobileNetV2(classes=2)
# load pretrain weights
model_weight_path = "mobilenet_v2.pth"
pre_weights = torch.load(model_weight_path)
# delete classifier weights
pre_dict = {k: v for k, v in pre_weights.items() if "classifier" not in k}
missing_keys, unexpected_keys = model.load_state_dict(pre_dict, strict=False)
# freeze features weights
for param in model.features.parameters():
    param.requires_grad = False

model.to(device)


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
train_loss_list = []  # 存储损失函数值
test_accuracy_list = []

start_time = datetime.datetime.now()
for epoch in range(epochs):
    model.train()
    epoch_loss = []
    epoch_test_accuracy = []

    for batch_idx, data in enumerate(train_data_loader, 1):
        # get the inputs, data is a list of [inputs, labels]
        x, y = data
        x = x.to(device)
        y = y.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        output_y = model(x)
        loss = criterion(output_y, y)
        loss.backward()
        optimizer.step()

        if loss < best_loss:
            best_model_weights = copy.deepcopy(model.state_dict())
            best_loss = loss

        with torch.no_grad():
            # 记录损失
            train_loss_list.append(loss.item())
            epoch_loss.append(loss.item())
            # test accuracy
            total = 0
            correct = 0
            for data in valid_data_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                predicted = torch.max(outputs.data, 1)[1]
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            test_accuracy = correct / total
            test_accuracy_list.append(test_accuracy)
            epoch_test_accuracy.append(test_accuracy)

    epoch_loss = np.mean(epoch_loss)  # 平均
    epoch_test_accuracy = np.mean(epoch_test_accuracy)
    print('step:' + str(epoch + 1) + '/' +
          str(epochs) + ' || Total Loss: %.4f' % (epoch_loss) +
          ' || Test accuracy: % .4f' % (epoch_test_accuracy))

SAVE_PATH = 'results/temp.pth'
torch.save(model.state_dict(), SAVE_PATH)
print('Finish Training.')

end_time = datetime.datetime.now()
print(f"训练耗时：{(end_time - start_time).seconds}" + "seconds")
print("Best loss: %.4f" % np.min(train_loss_list))
print("Best test accuracy: %.4f" % np.max(test_accuracy_list))

# %% 画图
# train loss
plt.plot(train_loss_list, label="train loss")
plt.title('Loss')
plt.legend()
plt.xlabel('Batches')
plt.ylabel('Loss')
plt.show()

plt.plot(test_accuracy_list, 'g', label="test accuracy")
plt.legend()
plt.title('Train and Test Accuracy')
plt.xlabel('Batches')
plt.ylabel('Accuracy')
plt.show()

# %%
# 4.评估模型，将自己认为最佳模型保存在 result 文件夹，其余模型备份在项目中其它文件夹，方便您加快测试通过。

model_path = 'results/temp.pth'
model = MobileNetV2(classes=2).to(device)
model.load_state_dict(torch.load(model_path))
correct = 0
total = 0
with torch.no_grad():
    for data in valid_data_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predicted = torch.max(outputs.data, 1)[1]
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the test images: %.2f %%' % (
    100 * correct / total))
