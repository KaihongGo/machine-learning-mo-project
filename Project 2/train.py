import os
import pickle
import time
import warnings
from datetime import timedelta

import jieba as jb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torchtext.data import BucketIterator, Dataset, Example, Field

from models.Net import Net

warnings.filterwarnings('ignore')


def processing_data(data_path, split_ratio=0.7):
    """
    数据处理
    :data_path：数据集路径
    :validation_split：划分为验证集的比重
    :return：train_iter,val_iter,TEXT.vocab 训练集、验证集和词典
    """
    # --------------- 已经实现好数据的读取，返回和训练集、验证集，可以根据需要自行修改函数 ------------------
    sentences = []  # 片段
    target = []  # 作者
    # 配置参数
    batch_size = 8
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    # 定义lebel到数字的映射关系
    labels = {'LX': 0, 'MY': 1, 'QZS': 2, 'WXB': 3, 'ZAL': 4}

    files = os.listdir(data_path)
    for file in files:
        if not os.path.isdir(file):
            f = open(data_path + "/" + file, 'r', encoding='UTF-8')  # 打开文件
            for index, line in enumerate(f.readlines()):
                sentences.append(line)
                target.append(labels[file[:-4]])

    mydata = list(zip(sentences, target))
    TEXT = Field(sequential=True, tokenize=lambda x: jb.lcut(x),
                 lower=True, use_vocab=True)
    LABEL = Field(sequential=False, use_vocab=False)
    FIELDS = [('text', TEXT), ('category', LABEL)]
    examples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS),
                        mydata))
    dataset = Dataset(examples, fields=FIELDS)
    TEXT.build_vocab(dataset)
    # save vocab
    with open('vocab.pkl', 'wb') as vocab:
        pickle.dump(TEXT.vocab, vocab)
    # 划分数据集
    train, val = dataset.split(split_ratio=split_ratio)
    # BucketIterator可以针对文本长度产生batch，有利于训练
    train_iter, val_iter = BucketIterator.splits(
        (train, val),  # 数据集
        batch_sizes=(batch_size, batch_size),
        device=device,  # 如果使用gpu，此处将-1更换为GPU的编号
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        repeat=False
    )
    # --------------------------------------------------------------------------------------------
    return train_iter, val_iter, TEXT.vocab


def get_time_diff(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def run(train_iter, val_iter, Text_vocab, save_model_path):
    """
    创建、训练和保存深度学习模型

    """
    # 配置
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    # --------------------- 实现模型创建、训练和保存等部分的代码 ---------------------
    vocab_size = len(Text_vocab)
    learning_rate = 1e-3
    num_epochs = 20
    require_improvement = 1000
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    # -----------------------------------------------------------------------------
    start_time = time.time()
    print("载入模型...")
    model = Net(vocab_size).to(device)
    print("模型载入完成...")
    time_diff = get_time_diff(start_time)
    print("Time usage:", time_diff)

    print("打印模型参数...")
    print(model.parameters)
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.shape)

    # 模型训练
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    val_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    stop = False  # 记录是否很久没有效果提升
    # plot
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    # 训练
    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        # scheduler.step() # 学习率衰减
        for i, batch in enumerate(train_iter):
            texts, labels = batch.text, batch.category
            outputs = model(texts)
            # model.zero_grad()
            optimizer.zero_grad()
            loss = F.cross_entropy(outputs, labels.long())
            # loss.backward(retain_graph = True)
            loss.backward()
            optimizer.step()

            if total_batch % 100 == 0:
                # 没多少轮输出在训练集和验证集上的效果
                labels = labels.cpu()
                predict = torch.argmax(outputs, 1).cpu()
                train_acc = metrics.accuracy_score(labels, predict)
                val_acc, val_loss = evaluate(model, val_iter)

                if val_loss < val_best_loss:
                    val_best_loss = val_loss
                    torch.save(model.state_dict(), save_model_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''

                time_diff = get_time_diff(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc,
                                 val_loss, val_acc, time_diff, improve))
                # plot mo platform
                # print('{{"metric": "Train Loss", "value": {}}}'.format(loss.item()))
                # print('{{"metric": "Train Acc", "value": {}}}'.format(train_acc))
                # print('{{"metric": "Val Loss", "value": {}}}'.format(val_loss))
                # print('{{"metric": "Val Acc", "value": {}}}'.format(val_acc))

                # plot
                train_loss_list.append(loss.item())
                train_acc_list.append(train_acc)
                val_loss_list.append(val_loss)
                val_acc_list.append(val_acc)
            total_batch = total_batch + 1
        #     if total_batch - last_improve > require_improvement:
        #         # 验证集loss超过1000batch没下降，结束训练
        #         print("No optimization for a long time, auto-stopping...")
        #         stop = True
        #         break
        # if stop:
        #     break
    # 保存模型（请写好保存模型的路径及名称）
    # torch.save(model.state_dict(), save_model_path)
    # 绘制曲线
    plt.figure(figsize=(15, 5.5))
    plt.subplot(121)
    plt.plot(train_acc_list, label='train acc')
    plt.plot(val_acc_list, label='val acc')
    plt.title("acc")
    plt.subplot(122)
    plt.plot(train_loss_list, label='train loss')
    plt.plot(val_loss_list, label='val loss')
    plt.title("loss")
    plt.legend()
    plt.savefig('results/results.jpg')
    # --------------------------------------------------------------------------------------------


def evaluate(model, data_iter):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            texts, labels = batch.text, batch.category
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss.item()
            labels = labels.data.cpu().numpy()
            predic = torch.argmax(outputs, 1).cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    acc = metrics.accuracy_score(labels_all, predict_all)
    model.train()
    return acc, loss_total / len(data_iter)


def main():
    """
    深度学习模型训练流程,包含数据处理、创建模型、训练模型、模型保存、评价模型等。
    如果对训练出来的模型不满意,你可以通过调整模型的参数等方法重新训练模型,直至训练出你满意的模型。
    如果你对自己训练出来的模型非常满意,则可以提交作业!
    :return:
    """
    data_path = "./dataset"  # 数据集路径
    save_model_path = "results/model.pth"  # 保存模型路径和名称
    train_val_split = 0.7  # 验证集比重

    # 获取数据、并进行预处理
    train_iter, val_iter, Text_vocab = processing_data(
        data_path, split_ratio=train_val_split)

    # 创建、训练和保存模型
    run(train_iter, val_iter, Text_vocab, save_model_path)

    # 评估模型
    # evaluate_mode(val_iter, save_model_path)


if __name__ == "__main__":
    main()
