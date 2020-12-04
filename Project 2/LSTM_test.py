###############################################################################
# 重要: 请务必把任务(jobs)中需要保存的文件存放在 results 文件夹内
# Important : Please make sure your files are saved to the 'results' folder
# in your jobs
###############################################################################
# 导入相关包
# %%
import copy
from marshal import load
import os
import pickle

import jieba as jb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
from torchtext import data, datasets
from torchtext.data import BucketIterator, Dataset, Example, Field


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
                target.append(labels[file[:-4]])  # 去掉.txt，如LX.txt

    return list(zip(sentences, target))


class Embedder(nn.Module):
    def __init__(self, TEXT, device, pretrain=True, embed_size=None):
        super(Embedder, self).__init__()
        # 加载预训练词向量
        # vocab = {}
        # with open('sgns.literature.word', 'r', encoding='utf-8') as f:
        #     lines = f.readlines()
        #     for line in lines[1:]:
        #         line = line.split()
        #         key = line[0]
        #         v=[]
        #         for s in line[1:]:  # 要先split，不然一整行是一个字符串
        #             v.append(float(s))
        #         vocab[key] = v

        # pickle.dump([vocab], open('vocab', 'wb'))
        if pretrain:
            vocab = pickle.load(open('vocab', 'rb'))  # 预训练词向量文件
            weight = torch.zeros(len(TEXT.vocab), 300)
            for i in range(len(TEXT.vocab)):
                weight[i] = torch.Tensor(vocab.get(TEXT.vocab.itos[i], weight[i]))
            
            self.embed = nn.Embedding.from_pretrained(weight, freeze=False).to(device)  # nn.Embedding.from_pretrained默认freeze为True，即不可训练的
        else:
            # nn.Embedding()本来就是可训练的
            self.embed = nn.Embedding(len(TEXT.vocab), embed_size)

    def forward(self, x):
        return self.embed(x)


class Net(nn.Module):
    def __init__(self, embedder, hidden_size=32, bidirectional=False, num_layers=1, mode='LSTM', attention=False, device='cuda'):
        super(Net, self).__init__()
        self.embedding = embedder
        self.hidden_size = hidden_size
        self.mode = mode
        self.attention = attention
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(self.embedding.embed.embedding_dim, self.hidden_size,
                          bidirectional=bidirectional, num_layers=num_layers)
        num_feat = self.hidden_size
        self.att_size = self.hidden_size
        
        if bidirectional:
            num_feat *= 2
            self.att_size = 2*self.att_size
        
        # attention中的query向量，可学习
        self.query = Variable(torch.zeros(self.att_size)
                              ).view(-1, 1).to(device)
        
        self.fc1 = nn.Linear(num_feat, 64)
        self.fc2 = nn.Linear(64, 5)
    # model:
    #     Net(
    #     (embedding): Embedder(
    #         (embed): Embedding(56261, 50)
    #     )
    #     (lstm): LSTM(50, 64, bidirectional=True)
    #     (fc1): Linear(in_features=128, out_features=64, bias=True)
    #     (fc2): Linear(in_features=64, out_features=5, bias=True)
    #     )

    def forward(self, x):
        """
        前向传播
        :param x: 模型输入
        :return: 模型输出
        """
        # input：(seq_len, batch, input_size)
        input = self.embedding(x)

        # hidden = (h_n, c_n),分别是LSTM每一层最后一个cell的hidden state和cell state，即短时记忆和长时记忆
        # output：(seq_len, batch, num_directions * hidden_size)，output同理，每一层的输出第一维依旧是batch
        # 是LSTM每一层（双向只算一层）所有cell的output vector（如果是双向则为拼接）
        output, hidden = self.lstm(input)
        if self.mode == 'LSTM':
            # h_n和c_n的shape: (num_layers * num_directions, batch_size, hidden_size)
            c_n = hidden[1]
        elif self.mode == 'LSTM+output':
            return output[-1]  # 用最后一个cell的output vector作为特征
        elif self.mode == 'GRU':  # GRU只有h_n
            c_n = hidden
        if not self.attention:
            # c_n.shape[1]=batch_size
            out = self.fc2(self.fc1(c_n.view(c_n.shape[1], -1)))
        else:  # 用attention作为输出，输出向量的加权求和
            shape = output.shape
            # 先过一层激活，(batch * seq_len, num_directions * hidden_size)
            M = f.tanh(output.view(-1, output.size()[2]))
            # 点积+softmax, (batch * seq_len, 1)
            att_weight = f.softmax(torch.mm(M, self.query), dim=1)
            att_weight = att_weight.view(shape[1], shape[0], 1)  # 最后增加一维是为了点乘
            # 转化为(batch, seq_len, num_directions * hidden_size)
            output = output.permute(1, 0, 2)
            # 点乘加权求和，第一维都是batch，左边的每个(seq_len, 1)都是一个句子的attention weight，点乘右边的该句子的输出矩阵
            out = (att_weight * output).sum(1)

        return out


if __name__ == "__main__":

    TEXT = Field(sequential=True, tokenize=lambda x: jb.lcut(x),
                 lower=True, use_vocab=True)
    LABEL = Field(sequential=False, use_vocab=False)
    FIELDS = [('text', TEXT), ('category', LABEL)]
    DATA_PATH = "dataset"
    mydata = load_data(DATA_PATH)
    examples = list(map(lambda x: Example.fromlist(
        list(x), fields=FIELDS), mydata))
    dataset = Dataset(examples, fields=FIELDS)
    TEXT.build_vocab(dataset)
    train, val = dataset.split(split_ratio=0.7)
    device = torch.device(
        "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    train_iter, val_iter = BucketIterator.splits(
        (train, val),  # 数据集
        batch_sizes=(8, 8),
        device=device,  # 如果使用gpu，此处将-1更换为GPU的编号
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        repeat=False
    )
    # 创建模型实例
    lr = 0.005
    epochs = 20
    emb_size = 50
    hidden_size = 64
    bi = True  # 不能用0和1
    num_layers = 1
    weight_decay = 5e-4
    attention = True

    model_save_path = './results/lstm.pth'

    embedder = Embedder(TEXT, device, False, emb_size)
    # 用词向量，但没用预训练的词向量时，随机初始化词向量，都是可训练的
    # embedder = Embedder(TEXT, device, True)  # 用预训练的词向量

    model = Net(embedder, hidden_size, bi, num_layers, mode='LSTM',
                attention=attention, device=device).to(device)
    print('model:\n', model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    print('start')
    train_acc_list, train_loss_list = [], []
    val_acc_list, val_loss_list = [], []
    best_acc = 0

    for epoch in range(epochs):
        epoch_loss = []
        for batch in train_iter:
            text, label = batch.text, batch.category
            text.to(device)
            label.to(device)
            optimizer.zero_grad()
            out = model(text)
            loss = loss_fn(out, label.long())
            loss.backward(retain_graph=True)
            optimizer.step()
            epoch_loss.append(loss.item())
        train_loss = np.mean(epoch_loss)
        train_loss_list.append(train_loss)

        with torch.no_grad():
            epoch_acc = []
            for batch in val_iter:
                text, label = batch.text, batch.category
                text.to(device)
                label.to(device)
                out = model(text)
                loss = loss_fn(out, label.long())
                acc = np.mean((torch.argmax(out, 1) == label).cpu().numpy())
                epoch_acc.append(acc)
        val_acc = np.mean(epoch_acc)
        val_acc_list.append(val_acc)

        if best_acc < val_acc:
            best_acc = val_acc
            print('best_acc:', best_acc)
            # 保存模型
            torch.save(model.state_dict(), model_save_path)

    # 绘制曲线
    plt.figure(figsize=(15, 5.5))
    plt.subplot(121)
    plt.plot(val_acc_list)
    plt.title("acc")
    plt.subplot(122)
    plt.plot(train_loss_list)
    plt.title("loss")

# %%
