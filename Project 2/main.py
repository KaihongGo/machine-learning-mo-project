# ==================  提交 Notebook 训练模型结果数据处理参考示范  ==================
# 导入相关包
import copy
import os
import pickle
import warnings

import jieba as jb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import BucketIterator, Dataset, Example, Field, Iterator

from models.Net import Net

warnings.filterwarnings('ignore')
# ------------------------------------------------------------------------------
# 本 cell 代码仅为 Notebook 训练模型结果进行平台测试代码示范
# 可以实现个人数据处理的方式，平台测试通过即可提交代码
#  -----------------------------------------------------------------------------


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


# 定义Field
TEXT = Field(sequential=True, tokenize=lambda x: jb.lcut(x),
             lower=True, use_vocab=True)
LABEL = Field(sequential=False, use_vocab=False)
FIELDS = [('text', TEXT), ('category', LABEL)]

# 构建中文词汇表
with open("vocab.pkl", 'rb') as vocab:
    TEXT.vocab = pickle.load(vocab)
# ----------------------------- 请加载您最满意的模型 -------------------------------
# 加载模型(请加载你认为的最佳模型)
# 加载模型,加载请注意 model_path 是相对路径, 与当前文件同级。
# 如果你的模型是在 results 文件夹下的 temp.pth 模型，则 model_path = 'results/temp.pth'

# 创建模型实例
vocab_size = len(TEXT.vocab)
model = Net(vocab_size)
model_path = "results/model.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# -------------------------请勿修改 predict 函数的输入和输出-------------------------


def predict(text):
    """
    :param text: 中文字符串
    :return: 字符串格式的作者名缩写
    """
    # ----------- 实现预测部分的代码，以下样例可代码自行删除，实现自己的处理方式 -----------
    labels = {0: 'LX', 1: 'MY', 2: 'QZS', 3: 'WXB', 4: 'ZAL'}
    # 自行实现构建词汇表、词向量等操作
    # 将句子做分词，然后使用词典将词语映射到他的编号
    text2idx = [TEXT.vocab.stoi[i] for i in jb.lcut(text)]
    # 转化为Torch接收的Tensor类型
    text2idx = torch.Tensor(text2idx).long()

    # 模型预测部分
    results = model(text2idx.view(-1, 1))
    prediction = labels[torch.argmax(results, 1).numpy()[0]]
    # --------------------------------------------------------------------------

    return prediction


if __name__ == "__main__":
    sen = "我听到一声尖叫，感觉到蹄爪戳在了一个富有弹性的东西上。定睛一看，不由怒火中烧。原来，趁着我不在，隔壁那个野杂种——沂蒙山猪刁小三，正舒坦地趴在我的绣榻上睡觉。我的身体顿时痒了起来，我的目光顿时凶了起来。"
    print(predict(sen))
