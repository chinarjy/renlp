#!/usr/bin/env python
# encoding: utf-8
"""
@author: jieye.ren
@contact: chinarjy@163.com
@software: pycharm
@file: cut_sentence.py
@time: 2021/12/10 1:02 PM
@desc: 分词
"""
# 系统库
import logging
import string
# 第三方库
import jieba
import jieba.posseg as psg
# 自定义库
import config

jieba.setLogLevel(logging.INFO)  # 不打印jieba日志
jieba.load_userdict(config.vocab_car_dict_path)  # 加载字典

letters = string.ascii_letters + '+0123456789'
filters = [',', '.', '-', '?', ' ', '/', '\\']
stopwords = [line.replace('\n', '') for line in open(config.stopwords_list_path, 'r', encoding='utf-8').readlines()]


def cut_sentence_by_word(sentence):
    """
    实现中英文分词（中文按单个字进行拆分）
    :param sentence:
    :return:
    """
    temp = ''
    result = []
    # python和c++哪个难？ --> [UI/UE, 和， c++， 哪， 个， 难， ？ ]
    for word in sentence:
        if word.lower() in letters:  # 字母
            temp += word.lower()
        else:  # 不是字母
            if temp != '':
                result.append(temp)
                temp = ''
            if word.strip() in filters:  # 标点符号
                continue
            else:  # 单子
                if len(word.strip()) != 0:
                    result.append(word.strip())
    if temp != '':
        result.append(temp)
    return result


def cut(sentence, by_word=False, use_stopwords=False, with_seg=False):
    """
    :param sentence: 句子
    :param by_word: 是否按单个字分词
    :param use_stopwords: 是否使用停用词
    :param with_sg: 是否返回词性
    :return:
    """
    assert by_word != True or with_seg != True
    if by_word:
        result = cut_sentence_by_word(sentence)
    else:
        n_result = []
        result = psg.lcut(sentence)
        for w in result:
            if len(w.word.strip()) == 0:
                continue
            if with_seg:
                n_result.append((w.word.strip(), w.flag))
            else:
                n_result.append(w.word.strip())
        result = n_result
    if use_stopwords and not with_seg:
        result_temp = [w for w in result if w not in stopwords]
        if len(result_temp) == 0:
            result_temp = [w for w in result]
        r_len = len(result_temp)
        assert r_len != 0, '{} 停用词过滤后句子长度为0'.format(sentence)
        return result_temp
    elif use_stopwords and with_seg:
        result_temp = []
        for w in result:
            if w[0] not in stopwords:
                result_temp.append((w[0], w[1]))
        if len(result_temp) == 0:
            for w in result:
                result_temp.append((w[0], w[1]))
        assert len(result_temp) != 0, '{} 停用词过滤后句子长度为0'.format(sentence)
        return result_temp
    return result
