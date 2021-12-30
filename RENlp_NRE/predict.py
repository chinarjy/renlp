# -*- coding = utf-8 -*-
# @Time :2021/10/29 15:52
# @Author:ren.jieye
# @Describe:数据预测
# @File : predict.py
# @Software: PyCharm IT
# 系统相关
import argparse
import os

# 框架相关
import torch
import torch.optim as optim
import torch.nn as nn

# 自定义
from RENlp_NRE.my_nre.config import config
from RENlp_NRE.my_nre.utils import make_seed, load_pkl, get_result
from RENlp_NRE.my_nre.process import split_sentences, bulid_data
from RENlp_NRE.my_nre import models


def predict(sent):
    __Models__ = {
        "CNN": models.CNN,
        "BiLSTM": models.BiLSTM,
        "BiLSTMPro": models.BiLSTMPro
    }

    parser = argparse.ArgumentParser(description='关系抽取')
    parser.add_argument('--model_name', type=str, default='CNN', help='model name')
    args = parser.parse_args()

    model_name = args.model_name if args.model_name else config.model_name

    # 计算设备设置
    if config.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda', config.gpu_id)
    else:
        device = torch.device('cpu')
    print(torch.cuda.is_available())
    #
    vocab_path = os.path.join(config.out_path, 'vocab.pkl')

    vocab = load_pkl(vocab_path, 'vocab')
    vocab_size = len(vocab.word2idx)
    model = __Models__[model_name](vocab_size, config)
    # print(model)
    #
    # optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=config.decay_rate,
    #                                                  patience=config.decay_patience)
    # loss_fn = nn.CrossEntropyLoss()

    model.load("model_file/Nre_CNN_epoch5_1126_10_49_10.pth")

    print("******************开始预测*********************")
    """
    # 已经识别出实体的前提下进行关系抽取，输入格式如下：
    # 孔正锡，导演，2005年以一部温馨的爱情电影《长腿叔叔》敲开电影界大门#长腿叔叔#影视作品#孔正锡#人物
    """
    text = sent
    data = text.split("#")
    entity1 = data[1]
    entity2 = data[3]
    head_index = data[0].index(entity1)
    tail_index = data[0].index(entity2)
    data.insert(3, str(head_index))
    data.insert(6, str(tail_index))
    raw_data = [data]
    new_test = split_sentences(raw_data)
    sents, head_pos, tail_pos, mask_pos = bulid_data(new_test, vocab)
    x = [torch.tensor(sents), torch.tensor(head_pos), torch.tensor(tail_pos), torch.tensor(mask_pos)]
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        y_pred = y_pred.argmax(dim=-1)
        result = get_result(entity1, entity2, y_pred.numpy()[0])
        print(result)


if __name__ == '__main__':
    predict('长春市有好多个5A景点，包括净月潭等#净月潭#景点#长春市#城市')
