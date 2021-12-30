#!/usr/bin/env python
# encoding: utf-8
"""
@author: jieye.ren
@contact: chinarjy@163.com
@software: pycharm
@file: application.py
@time: 2021/12/3 12:09 PM
@desc: 使用结巴和基于规则的分词（自己写的）进行分词
"""
from BMM import BMM
from RMM import RMM
from MM import MM
import jieba


def sent2word(sentence):
    """
    Segment a sentence to words
    Delete stopwords
    """
    jieba.load_userdict('../data/一汽大众领域词典.txt')  # 定制自己的领域词典，用以提升分词效果

    segList = jieba.cut(sentence, cut_all=False)  # 默认为False：精确分词，cut_all=True：全模式（所有可能都打印出来）
    segResult = []
    for w in segList:
        segResult.append(w)

    stopwords = open('../data/中文停用词词典.txt', 'r', encoding='UTF-8').read().splitlines()
    newSent = []
    for word in segResult:
        if word in stopwords:
            # print "stopword: %s" % word
            continue
        else:
            newSent.append(word)
    return newSent


if __name__ == '__main__':
    sentence = '中国人民解放军于1949年7月通过了南京长江大桥，打败了国民党守军，收复南京、上海等城市'
    # jieba分词
    print('jieba Word_seg result : {}'.format(list(jieba.cut(sentence))))
    # 正向最大匹配法
    tokenizer_mm = MM()
    print('RENlp MM Word_seg result : {}'.format(tokenizer_mm.cut(sentence)))
    # 逆向最大匹配法
    tokenizer_rmm = RMM()
    print('RENlp RMM Word_seg result : {}'.format(tokenizer_rmm.cut(sentence)))
    # 双向最大匹配法
    tokenizer_bmm = BMM()
    print('RENlp BMM Word_seg result : {}'.format(tokenizer_bmm.cut(sentence)))

