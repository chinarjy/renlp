#!/usr/bin/env python
# encoding: utf-8
"""
@author: jieye.ren
@contact: chinarjy@163.com
@software: pycharm
@file: build_corpus.py
@time: 2021/12/29 9:05
@desc: 将夜小说语料处理
"""
from Utils import cut, file_exist, logger
from tqdm import tqdm


def cut_txt():
    """
    处理将夜小说语料
    """
    jiangye_origin_path = 'RENlp_word2vec/data/origin/将夜.txt'
    lines = open(jiangye_origin_path, 'r', encoding='utf-8').readlines()
    jiangye_corpus_path = 'RENlp_word2vec/data/corpus/jiangye_corpus.txt'
    file_exist(jiangye_corpus_path)
    jiangye_corpus_file = open(jiangye_corpus_path, 'a', encoding='utf-8')
    logger.info('创建jiangye_corpus.txt')

    for line in tqdm(lines, desc='将夜小说语料处理'):
        if len(line.strip()) == 0:
            continue
        sen = cut(line.strip(), by_word=False, use_stopwords=False, with_seg=False)
        jiangye_corpus_file.write(' '.join(sen) + '\n')
    jiangye_corpus_file.close()


def prepare_train_corpus():
    corpus_path = 'RENlp_word2vec/data/train.txt'
    train_corpus_path = 'RENlp_word2vec/data/train_corpus.txt'
    file_exist(train_corpus_path)
    train_corpus_file = open(train_corpus_path, 'a', encoding='utf-8')
    for line in tqdm(open(corpus_path, 'r', encoding='utf-8').readlines(), desc='训练语料处理'):
        line = cut(line.split('***')[0])
        line = ' '.join(line) + '\n'
        train_corpus_file.write(line)
    logger.info('{} 保存成功'.format(train_corpus_path))
    train_corpus_file.close()
