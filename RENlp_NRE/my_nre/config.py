# -*- coding = utf-8 -*-
# @Time :2021/10/27 11:21
# @Author:ren.jieye
# @Describe:配置类
# @File : config.py
# @Software: PyCharm IT

"""
模型配置类
"""


class Config(object):
    model_name = "CNN"  # 'CNN','BiLSTM','BiLSTMPro'
    data_path = 'data/out/vocab.pkl'  # 原始数据保存路径
    out_path = 'data/out'  # 数据处理结果保存路径
    is_CN = True  # 是否是中文
    word_segment = True  # 是否分词
    trimed = False  # False处理低频次  True不处理
    relation_type = 10  # 关系种类
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

    epoch = 10

    train_log = True
    log_interval = 10

    f1_norm = ['macro', 'micro']

    # CNN
    out_channels = 100
    kernel_size = [3, 5]

    # BiLSTM
    lstm_layers = 2

    # 初始化种子
    seed = 1

    use_gpu = True
    gpu_id = 0


config = Config()
