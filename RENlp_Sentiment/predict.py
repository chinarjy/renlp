#!/usr/bin/env python
# encoding: utf-8
"""
@author: jieye.ren
@contact: chinarjy@163.com
@software: pycharm
@file: predict.py
@time: 2021/12/4 9:27 PM
@desc: 情感预测
"""

# 系统相关
import argparse
import os
import codecs
import time
import warnings
warnings.filterwarnings('ignore')
# 框架相关
import torch
from RENlp_Sentiment.logger import logger

# 自定义
from RENlp_Sentiment.config import config
from RENlp_Sentiment.utils.utils import load_pkl, get_result, get_result_eat
from RENlp_Sentiment.utils.process import split_sentences, bulid_data
import models


def load_model():
    __Models__ = {
        "CNN": models.Sentiment_CNN
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

    out_path = ''
    print(torch.cuda.is_available())
    if config.type == 'eat':
        out_path = config.out_path1
    elif config.type == 'car':
        out_path = config.out_path
    vocab_path = os.path.join(out_path, 'vocab.pkl')

    vocab = load_pkl(vocab_path, 'vocab')
    vocab_size = len(vocab.word2idx)
    model = __Models__[model_name](vocab_size, config)

    model.load("model_file/car_Sentiment_CNN_epoch2_1206_11_33_56.pth")
    model.to(device)
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
        if config.type == 'car':
            result = get_result(y_pred.numpy()[0])
        elif config.type == 'eat':
            result = get_result_eat(y_pred.numpy()[0])
        print('未鉴权预测tag : {}, 得分: {}'.format(result.get('tag'), score))
        if score > config.min_score:
            result = result.get('tag')
        else:
            result = 'None'
        return result, score


if __name__ == '__main__':
    # 模型和字典下载
    model, vocab = load_model()
    parser = argparse.ArgumentParser(description='情感分析')  # 创建解析器
    parser.add_argument('--sen_pre', type=bool, default=True, help='测试方式：True-单句测试， False-文档测试')  # 添加参数
    args = parser.parse_args()  # 解析参数
    predict_way = args.sen_pre

    if predict_way:
        print('************开始单句测试*******************')
        while(True):
            print('请输入句子，输入0结束：')
            text = input()
            if text == '0':
                break
            predict_intent, score = predict(text, model, vocab)
            print('鉴权后预测tag： {}'.format(predict_intent))
    else:
        print('***************开始批量测试*********************')
        logger.info('***************开始批量测试*********************')
        p_tags = []
        t_tags = []
        with codecs.open('data/origin/test.txt', 'r', encoding='utf-8') as test_f:
            lines = test_f.readlines()
            result = ''
            for line in lines:
                sentence = line.strip().split('***')[0]
                t_tag = line.strip().split('***')[1]
                print('****************第{}个句子开始****************'.format(lines.index(line)))
                print('sentence : {} , tag : {}'.format(sentence, t_tag))
                p_tag, score = predict(sentence, model, vocab)
                t_tags.append(t_tag)
                p_tags.append(p_tag)
                # 保存预测结果
                result += sentence + '***' + t_tag + '***' + p_tag + '\r'
                if p_tag != t_tag:
                    logger.info('预测结果与实际标注结果不一致： 【{}】, true_tag : 【{}】 , pred_tag : 【{}】'.format(sentence, t_tag, p_tag))
                print('tag : {}'.format(p_tag))
                print('****************第{}个句子结束****************'.format(lines.index(line)))
        test_f.close()
        with codecs.open('data/test_result/test_result_cnn_{}_{}.txt'.format(config.type, time.strftime('%m%d_%H_%M_%S.pth')), 'w',
                         encoding='utf-8') as result_file:
            result_file.write(result)
        result_file.close()

        from sklearn.metrics import classification_report
        report = classification_report(t_tags, p_tags)
        logger.info(report)
        logger.info('*****************批量测试结束************************')



