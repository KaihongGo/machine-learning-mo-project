import os
from os.path import split
import numpy as np
from numpy import random
from numpy.core.numeric import indices


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
            for line in f:
                line = line.rstrip('\n')
                sentences.append(line)
                target.append(labels[file[:-4]])

    return list(zip(sentences, target))


data_path = "dataset"
dataset = load_data(data_path)
dataset = np.array(dataset)

validation_split = 0.2
shuffle_dataset = True
random_seed = 42

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split*dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]

train_dataset, val_dataset = dataset[train_indices], dataset[val_indices]

print(train_dataset.size)
print(val_dataset.size)

with open('train.txt', 'w', encoding='UTF-8') as train:
    for data in train_dataset:
        train.write(data[0] + "\t" + str(data[1]) + '\n')


with open('dev.txt', 'w', encoding='UTF-8') as train:
    for data in val_dataset:
        train.write(data[0] + "\t" + str(data[1]) + '\n')


with open('test.txt', 'w', encoding='UTF-8') as train:
    for data in val_dataset:
        train.write(data[0] + "\t" + str(data[1]) + '\n')
