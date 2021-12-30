# -*- coding = utf-8 -*-
# @Time :2021/9/24 11:09
# @Author:ren.jieye
# @Describe:
# @File : stopwords.py
# @Software: PyCharm IT

def remove_stopwords(segResult):
    stopwords = open('data/中文停用词词典.txt', 'r', encoding='UTF-8').read().splitlines()
    newSent = []
    for word in segResult:
        if word in stopwords:
            # print "stopword: %s" % word
            continue
        else:
            newSent.append(word)
    return newSent


class Stopwords:
    pass
