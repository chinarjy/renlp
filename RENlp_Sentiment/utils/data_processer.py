#!/usr/bin/env python
# encoding: utf-8
"""
@author: jieye.ren
@contact: chinarjy@163.com
@software: pycharm
@file: data_processer.py
@time: 2021/12/3 2:42 PM
@desc: 数据处理
"""
import codecs
import csv
import pandas as pd
from RENlp_Sentiment.utils.utils import save_pkl, ensure_dir


def load_file(filepath: str, type: str):
    labels = []
    sentences = []
    if type == 'csv':
        train_file = codecs.open('data/origin/train.txt', 'w', encoding='utf-8')
        test_file = codecs.open('data/origin/test.txt', 'w', encoding='utf-8')
        label_file = codecs.open('data/origin/tag.txt', 'w', encoding='utf-8')

        df = pd.read_csv(filepath, encoding='utf-8')
        sentences = df['comment'].values.tolist()
        labels = df['emo'].values.tolist()
    elif type == 'txt':
        train_file = codecs.open('data/origin1/train.txt', 'w', encoding='utf-8')
        test_file = codecs.open('data/origin1/test.txt', 'w', encoding='utf-8')
        label_file = codecs.open('data/origin1/tag.txt', 'w', encoding='utf-8')

        with codecs.open('data/origin1/dataset1.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                sentence = line.strip().split(',', 1)[1]
                label = line.strip().split(',', 1)[0]
                label = 'positive' if label == '1' else 'negative'
                sentences.append(sentence)
                labels.append(label)

    # 构建训练集和测试集
    labellist = set(labels)
    train_text = ''
    test_text = ''
    tag_text = ''
    total_num = len(sentences)
    train_num = int(total_num * 0.8)
    for i, line in enumerate(sentences):
        if i < train_num:
            train_text += line + '***' + labels[i] + '\n'
        else:
            test_text += line + '***' + labels[i] + '\n'
    for tag in labellist:
        tag_text += tag + '\n'
    train_file.write(train_text)
    test_file.write(test_text)
    label_file.write(tag_text)


if __name__ == '__main__':
    load_file('../data/origin1/dataset1.txt')
