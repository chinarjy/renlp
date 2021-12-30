# -*- coding = utf-8 -*-
# @Time :2021/9/24 11:00
# @Author:ren.jieye
# @Describe:jieba分词+停用词过滤
# @File : segByJieba.py
# @Software: PyCharm IT
import jieba


def sent2word(sentence):
    """
    Segment a sentence to words
    Delete stopwords
    """
    # jieba.load_userdict('../data/一汽大众领域词典.txt')  # 定制自己的领域词典，用以提升分词效果

    segList = jieba.cut(sentence, cut_all=False)  # 默认为False：精确分词，cut_all=True：全模式（所有可能都打印出来）
    segResult = []
    for w in segList:
        segResult.append(w)

    stopwords = open('data/中文停用词词典.txt', 'r', encoding='UTF-8').read().splitlines()
    newSent = []
    for word in segResult:
        if word in stopwords:
            # print "stopword: %s" % word
            continue
        else:
            newSent.append(word)
    return newSent


class Segmentation:
    pass
