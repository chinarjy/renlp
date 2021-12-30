#!/usr/bin/env python
# encoding: utf-8
"""
@author: jieye.ren
@contact: chinarjy@163.com
@software: pycharm
@file: config.py
@time: 2021/12/3 8:59 AM
@desc: 模型配置类
"""


class Config(object):
    model_name = "CNN"  # 'CNN','BiLSTM','BiLSTMPro'
    type = "car"  # car or eat
    data_path = 'data/origin'  # 原始数据保存路径
    data_path1 = 'data/origin1'
    out_path = 'data/output'  # 数据处理结果保存路径
    out_path1 = 'data/output1'
    is_CN = True  # 是否是中文
    word_segment = True  # 是否分词
    trimed = False  # False处理低频次  True不处理
    relation_type = 3  # 情感种类
    min_freq = 2  # 低频词处理
    #  位置编码
    pos_limit = 50  # [-50,50]
    pos_size = 102  # 2*pos_limit+2  位置编号在0-101之间

    word_dim = 300
    pos_dim = 10

    hidden_size = 100  # FC连接数
    dropout = 0.5

    batch_size = 128

    learning_rate = 0.001  # 超参

    decay_rate = 0.3  # 衰减率

    decay_patience = 5  # 次数

    epoch = 30

    train_log = True
    log_interval = 1

    f1_norm = ['macro', 'micro']

    # CNN
    out_channels = 100
    kernel_size = [3, 5]

    min_score = 0

    # BiLSTM
    lstm_layers = 2

    # 初始化种子
    seed = 1

    use_gpu = True
    gpu_id = 0


config = Config()
