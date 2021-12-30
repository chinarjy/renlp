#!/usr/bin/env python
# encoding: utf-8
"""
@author: jieye.ren
@contact: chinarjy@163.com
@software: pycharm
@file: word2vec_config.py
@time: 2021/12/29 9:53
@desc: word2vec 配置类
"""
# word2vec model 参数
sg = 1  # 算法选择  默认sg=0： CBOW算法 , sg=1 : skip_gram算法，对低频次敏感
vector_size = 100 # 神经网络层数，一般取100-200之间
window = 5  # 窗口大小
min_count = 2  # 词频过滤， 小于min_count的词会被过滤掉
negative = 3  # 如果大于0，则会采用negativesamping，用于设置多少个noise words
sample = 0.001  # 表示更高频率的词被随机下采样到所设置的阈值
hs = 1  # 1 表示softmax将会被使用，hs=0且negative不为0，则负采样将会被选择使用
workers = 4  # 线程数
