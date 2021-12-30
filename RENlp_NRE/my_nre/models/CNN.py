# -*- coding = utf-8 -*-
# @Time :2021/10/27 11:22
# @Author:ren.jieye
# @Describe:
# @File : CNN.py
# @Software: PyCharm IT
import torch
import torch.nn as nn
import torch.nn.functional as F
from RENlp_NRE.my_nre.models import BasicModule, Embedding


class CNN(BasicModule):
    def __init__(self, vocab_size, config):
        super(CNN, self).__init__()
        self.model_name = 'Nre_CNN'
        self.out_channels = config.out_channels
        self.kernel_size = config.kernel_size
        self.word_dim = config.word_dim  # 词向量维度
        self.pos_size = config.pos_size
        self.pos_dim = config.pos_dim  # 位置向量的维度
        self.hidden_size = config.hidden_size
        self.dropout = config.dropout
        self.out_dim = config.relation_type  # 输出的维度实际就是关系种类的数量

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size]
        for k in self.kernel_size:
            assert k % 2 == 1

        # 嵌入层：tensor类的矩阵送入神经网络
        self.embedding = Embedding(vocab_size, self.word_dim, self.pos_size, self.pos_dim)

        self.input_dim = self.word_dim + self.pos_dim * 2

        # 卷积层
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.input_dim,
                      out_channels=self.out_channels,
                      kernel_size=k,
                      padding=k//2,
                      bias=None) for k in self.kernel_size
        ])

        self.conv_dim = len(self.kernel_size) * self.out_channels
        self.fc1 = nn.Linear(self.conv_dim, self.hidden_size)  # 线性转换
        self.dropout = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(self.hidden_size, self.out_dim)

    def forward(self, input):
        """
        :param input: word_ids, headpos,tailpos,mask
        :return:
        """
        *x, mask = input

        # 嵌入层
        # (N, C, L):N is a batch size, CC denotes a number of channels, LL is a length of signal sequence.
        x = self.embedding(x)  # 三维数据 [128,124,320]  将一个长度为3的tensor类型list转化为3维的tensor
        x = torch.transpose(x, 1, 2)  # [128,320,124]:转换成要求的矩阵格式

        # 卷积层
        x = [F.leaky_relu(conv(x)) for conv in self.convs]
        x = torch.cat(x, dim=1)
        s_len = x.size(-1)
        # 池化
        x = F.max_pool1d(x, s_len)
        x = x.squeeze(-1)

        '''
        每次训练都随机让一定神经元停止参与运算,防止过拟合
        提高模型稳定性和鲁棒性
        '''
        x = self.dropout(x)  # 通过阻止特征检测器的共同作用来提高神经网络的性能

        # 全连接层，分类器
        x = F.leaky_relu(self.fc1(x))  # LeakyReLU(x) = max(0, x) + negative_slope * min(0, x)  ; negative_slope = 0.01
        x = F.leaky_relu(self.fc2(x))  # 激活函数：增加模型的非线性表达能力
        return x
