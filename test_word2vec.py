#!/usr/bin/env python
# encoding: utf-8
"""
@author: jieye.ren
@contact: chinarjy@163.com
@software: pycharm
@file: test_word2vec.py
@time: 2021/12/28 15:34
@desc: word2vec 代码测试文件
"""
from RENlp_word2vec.prepare_corpus import cut_txt, prepare_train_corpus
from RENlp_word2vec import build_word2vec_model, get_vector, most_similar, similarity

if __name__ == '__main__':
    # 处理训练语料（微粒贷语料库）
    # prepare_train_corpus()
    # 将小说原文处理、分词并保存
    # cut_txt()
    # 构建模型
    # build_word2vec_model()
    # *******************获取词向量**********************
    # words = ['宁缺', '君陌']
    # print(get_vector(words, binary=True))
    # print(get_vector(words))
    # *******************获取相似词**********************
    # print(most_similar('李慢慢', topk=5))
    # *******************比较两个词的相似度****************
    print('相似度: ', similarity('宁缺', '君陌'))
