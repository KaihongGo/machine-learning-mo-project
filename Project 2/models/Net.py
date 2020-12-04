import torch
import torch.nn as nn


class Net(nn.Module):
    """
    Recurrent Neural Network for Text Classification with Multi-Task Learning
    """

    def __init__(self, vocab_size):
        """
        docstring
        """
        self.embedding_pretrained = None
        self.vocab_size = vocab_size
        self.embed_size = 200
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.5
        self.num_classes = 5

        super(Net, self).__init__()
        # if self.embedding_pretrained is not None:
        #     self.embedding = nn.Embedding.from_pretrained(
        #         self.embedding_pretrained, freeze=False)
        # else:
        self.embedding = nn.Embedding(
            self.vocab_size, self.embed_size, padding_idx=self.vocab_size - 1)

        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers,
                            bidirectional=True, batch_first=True, dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_size*2, self.num_classes)

        # init model parameters
        method = 'xavier'
        exclude = 'embedding'
        for name, w in self.named_parameters():
            if exclude not in name:
                if 'weight' in name:
                    if method == 'xavier':
                        nn.init.xavier_normal_(w)
                    elif method == 'kaiming':
                        nn.init.kaiming_normal_(w)
                    else:
                        nn.init.normal_(w)
                elif 'bias' in name:
                    nn.init.constant_(w, 0)
                else:
                    pass

    def forward(self, x):
        """
        docstring
        """
        x = torch.transpose(x, 0, 1)
        # [batch_size, seq_len, embeding]=[128, 32, 300]
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out
