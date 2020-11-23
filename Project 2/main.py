# %%
# 导入相关包
import copy
import os
import random

import jieba as jb
import jieba.analyse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init
from torchtext import data, datasets
from torchtext.data import BucketIterator, Dataset, Example, Field, Iterator

# %%
# 读取数据集，保存在字典中
dataset = {}
path = "dataset/"
files = os.listdir(path)
for file in files:
    if not os.path.isdir(file) and not file[0] == '.':  # 跳过隐藏文件和文件夹
        f = open(path+"/"+file, 'r',  encoding='UTF-8')  # 打开文件
        for line in f.readlines():
            dataset[line] = file[:-4]

# %%
# 展示数据
name_zh = {'LX': '鲁迅', 'MY': '莫言', 'QZS': '钱钟书', 'WXB': '王小波', 'ZAL': '张爱玲'}
for (k, v) in list(dataset.items())[:6]:
    print(k, '---', name_zh[v])

# %%
# jieba演示
# 精确模式分词
titles = [".".join(jb.cut(t, cut_all=False)) for t, _ in dataset.items()]
print("精确模式分词结果:\n", titles[0])
# 全模式分词
titles = [".".join(jb.cut(t, cut_all=True)) for t, _ in dataset.items()]
print("全模式分词结果:\n", titles[0])
# 搜索引擎模式分词
titles = [".".join(jb.cut_for_search(t)) for t, _ in dataset.items()]
print("搜索引擎模式分词结果:\n", titles[0])


# %%
# 将片段进行词频统计
str_full = {}
str_full['LX'] = ""
str_full['MY'] = ""
str_full['QZS'] = ""
str_full['WXB'] = ""
str_full['ZAL'] = ""

for (k, v) in dataset.items():
    str_full[v] += k

for (k, v) in str_full.items():
    print(k, ":")
    for x, w in jb.analyse.extract_tags(v, topK=5, withWeight=True):
        print('%s %s' % (x, w))

# %%


def load_data(path):
    """
    读取数据和标签
    :param path:数据集文件夹路径
    :return:返回读取的片段和对应的标签
    """
    sentences = []  # 片段
    target = []  # 作者

    # 定义lebel到数字的映射关系
    labels = {'LX': 0, 'MY': 1, 'QZS': 2, 'WXB': 3, 'ZAL': 4}

    files = os.listdir(path)
    for file in files:
        if not os.path.isdir(file):
            f = open(path + "/" + file, 'r', encoding='UTF-8')  # 打开文件
            for index, line in enumerate(f.readlines()):
                sentences.append(line)
                target.append(labels[file[:-4]])

    return list(zip(sentences, target))


# %%
# 定义Field
TEXT = Field(sequential=True, tokenize=lambda x: jb.lcut(x),
             lower=True, use_vocab=True)
LABEL = Field(sequential=False, use_vocab=False)
FIELDS = [('text', TEXT), ('category', LABEL)]

# 读取数据，是由tuple组成的列表形式
mydata = load_data(path)

# 使用Example构建Dataset
examples = list(map(lambda x: Example.fromlist(
    list(x), fields=FIELDS), mydata))
dataset = Dataset(examples, fields=FIELDS)
# 构建中文词汇表
TEXT.build_vocab(dataset)
# %%
# 切分数据集
train, val = dataset.split(split_ratio=0.7)
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# 生成可迭代的mini-batch
train_iter, val_iter = BucketIterator.splits(
    (train, val),  # 数据集
    batch_sizes=(8, 8),
    device=device,  # 如果使用gpu，此处将-1更换为GPU的编号
    sort_key=lambda x: len(x.text),
    sort_within_batch=False,
    repeat=False
)
# %%
# Pytorch定义模型的方式之一：
# 继承 Module 类并实现其中的forward方法


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = torch.nn.LSTM(1, 64)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        """
        前向传播
        :param x: 模型输入
        :return: 模型输出
        """
        output, hidden = self.lstm(x.unsqueeze(2).float())
        h_n = hidden[1]
        out = self.fc2(self.fc1(h_n.view(h_n.shape[1], -1)))
        return out


# 创建模型实例
model = Net().to(device)

# %%
# 查看模型参数
for name, parameters in model.named_parameters():
    print(name, ':', parameters.size())

# %%
# 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# %%
# 模型训练
train_acc_list, train_loss_list = [], []
val_acc_list, val_loss_list = [], []
for epoch in range(3):
    train_acc, train_loss = 0, 0
    val_acc, val_loss = 0, 0
    for idx, batch in enumerate(train_iter):
        text, label = batch.text, batch.category
        optimizer.zero_grad()
        out = model(text)
        loss = loss_fn(out, label.long())
        loss.backward(retain_graph=True)
        optimizer.step()
        accracy = np.mean((torch.argmax(out, 1) == label).cpu().numpy())
        # 计算每个样本的acc和loss之和
        train_acc += accracy*len(batch)
        train_loss += loss.item()*len(batch)

        print("\r opech:{} loss:{}, train_acc:{}".format(
            epoch, loss.item(), accracy), end=" ")

    # 在验证集上预测
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text, label = batch.text, batch.category
            out = model(text)
            loss = loss_fn(out, label.long())
            accracy = np.mean((torch.argmax(out, 1) == label).cpu().numpy())
            # 计算一个batch内每个样本的acc和loss之和
            val_acc += accracy*len(batch)
            val_loss += loss.item()*len(batch)

    train_acc /= len(train_iter.dataset)
    train_loss /= len(train_iter.dataset)
    val_acc /= len(val_iter.dataset)
    val_loss /= len(val_iter.dataset)
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)
    val_acc_list.append(val_acc)
    val_loss_list.append(val_loss)

# 保存模型
torch.save(model.state_dict(), 'results/temp.pth')

# 绘制曲线
plt.figure(figsize=(15, 5.5))
plt.subplot(121)
plt.plot(train_acc_list)
plt.plot(val_acc_list)
plt.title("acc")
plt.subplot(122)
plt.plot(train_loss_list)
plt.plot(val_loss_list)
plt.title("loss")

# %%
# 加载模型并预测
model = Net()
model_path = "results/temp.pth"
model.load_state_dict(torch.load(model_path))
print('模型加载完成...')

# %%
# 这是一个片段
text = "中国中流的家庭，教孩子大抵只有两种法。其一是任其跋扈，一点也不管，\
    骂人固可，打人亦无不可，在门内或门前是暴主，是霸王，但到外面便如失了网的蜘蛛一般，\
    立刻毫无能力。其二，是终日给以冷遇或呵斥，甚于打扑，使他畏葸退缩，彷佛一个奴才，\
    一个傀儡，然而父母却美其名曰“听话”，自以为是教育的成功，待到他们外面来，则如暂出樊笼的\
    小禽，他决不会飞鸣，也不会跳跃。"

labels = {0: '鲁迅', 1: '莫言', 2: '钱钟书', 3: '王小波', 4: '张爱玲'}

# 将句子做分词，然后使用词典将词语映射到他的编号
text2idx = [TEXT.vocab.stoi[i] for i in jb.lcut(text)]

# 转化为Torch接收的Tensor类型
text2idx = torch.Tensor(text2idx).long()

# 预测
print(labels[torch.argmax(model(text2idx.view(-1, 1)), 1).numpy()[0]])

# %%
