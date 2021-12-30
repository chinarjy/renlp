#!/usr/bin/env python
# encoding: utf-8
"""
@author: jieye.ren
@contact: chinarjy@163.com
@software: pycharm
@file: predict.py
@time: 2021/12/3 8:58 AM
@desc: Intent预测-模型：Intent_CNN_epoch24_1203_10_58_26.pth
"""

# 系统相关
import argparse
import os
import codecs
import warnings
warnings.filterwarnings('ignore')
# 框架相关
import torch

# 自定义
from RENlp_Intent.config import config
from utils.utils import load_pkl, get_result
from utils.process import split_sentences, bulid_data
import models


def load_model():
    __Models__ = {
        "CNN": models.CNN,
        # "BiLSTM": models.BiLSTM,
        # "BiLSTMPro": models.BiLSTMPro
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

    model.load("model_file/Intent_CNN_epoch24_1203_10_58_26.pth")
    return model, vocab


def predict(sent, model, vocab):
    """
    # ID.4 CROZZ采用30色贯穿式氛围灯和30色透光饰条，颜色可以自主选择，也可以与驾驶模式和信息娱乐大屏联动。
    """
    text = sent
    raw_data = [text]
    new_test = split_sentences(raw_data)
    sents = bulid_data(new_test, vocab)
    x = torch.tensor(sents)
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        score_list = y_pred.numpy()[0]
        score = max(score_list)
        y_pred = y_pred.argmax(dim=-1)
        result = get_result(y_pred.numpy()[0])
        print('未鉴权预测tag : {}, 得分: {}'.format(result.get('tag'), score))
        if score > config.min_score:
            result = result.get('tag')
        else:
            result = 'None'
        return result, score


if __name__ == '__main__':
    # 模型和字典下载
    model, vocab = load_model()
    parser = argparse.ArgumentParser(description='意图识别')  # 创建解析器
    parser.add_argument('--sen_pre', type=bool, default=True, help='测试方式：True-单句测试， False-文档测试')  # 添加参数
    args = parser.parse_args()  # 解析参数
    predict_way = args.sen_pre

    if predict_way:
        print('************开始单句测试*******************')
        predict_intent, score = predict('后边的摄像头，随着你的方向盘转动，可以转动不光线能转，你看着摄像头扩大你的视野', model, vocab)
        print('鉴权后预测tag： {}'.format(predict_intent))
    else:
        print('***************开始批量测试*********************')
        p_tags = []
        t_tags = []
        with codecs.open('data/origin/test.txt', 'r', encoding='utf-8') as test_f:
            lines = test_f.readlines()
            for line in lines:
                sentence = line.strip().split('***')[0]
                t_tag = line.strip().split('***')[1]
                print('****************第{}个句子开始****************'.format(lines.index(line)))
                print('sentence : {} , tag : {}'.format(sentence, t_tag))
                p_tag, score = predict(sentence, model, vocab)
                t_tags.append(t_tag)
                p_tags.append(p_tag)
                # 保存预测结果
                with codecs.open('data/test_result/test_result_cnn.txt', 'a', encoding='utf-8') as result_file:
                    result_file.write(sentence + '***' + t_tag + '***' + p_tag + '\r')
                result_file.close()
                print('tag : {}'.format(p_tag))
                print('****************第{}个句子结束****************'.format(lines.index(line)))
        test_f.close()
        from sklearn.metrics import classification_report

        print(classification_report(t_tags, p_tags))

