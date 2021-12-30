#!/usr/bin/env python
# encoding: utf-8
"""
@author: jieye.ren
@contact: chinarjy@163.com
@software: pycharm
@file: tag_jieba.py
@time: 2021/12/3 1:13 PM
@desc: 使用jieba进行词性标注
"""
import jieba
import jieba.posseg as psg

# jieba.load_userdict('data/一汽大众领域词典.txt')


def add_word(words):
    jieba.add_word('一汽-大众')
    pass


def posseg(sentence):
    seg_list = psg.cut(sentence)
    return seg_list


# add_word('yi')
seg_lst = posseg('我是一汽-大众的员工')
print(list(seg_lst))
print(list(jieba.cut('我是一汽-大众的员工')))
