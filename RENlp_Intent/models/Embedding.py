# -*- coding = utf-8 -*-
# @Time :2021/10/27 14:39
# @Author:ren.jieye
# @Describe:
# @File : Embedding.py
# @Software: PyCharm IT
import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size, word_dim, pos_size, pos_dim):
        super(Embedding, self).__init__()
        self.word_embed = nn.Embedding(vocab_size, word_dim, padding_idx=0)

    def forward(self, x):
        words = x
        word_embed = self.word_embed(words)
        feature_embed = torch.cat([word_embed], dim=-1)  # 拼接
        return feature_embed
