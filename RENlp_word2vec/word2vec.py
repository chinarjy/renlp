#!/usr/bin/env python
# encoding: utf-8
"""
@author: jieye.ren
@contact: chinarjy@163.com
@software: pycharm
@file: word2vec.py
@time: 2021/12/28 15:09
@desc: word2vec
"""
import gensim
from gensim.models import word2vec, KeyedVectors
from Utils import logger
import os
import RENlp_word2vec.word2vec_config as config

corpus_path = 'RENlp_word2vec/data/corpus/jiangye_corpus.txt'
word2vec_model_path = 'RENlp_word2vec/models/word2vec_model.model'
word2vec_binary_model_path = 'RENlp_word2vec/models/word2vec_bin_model.bin'


def build_word2vec_model():
    """
    构建word2vec模型
    """
    sens = word2vec.Text8Corpus(corpus_path)
    model = gensim.models.Word2Vec(sens, sg=1, vector_size=config.vector_size,
                                   window=config.window, min_count=config.min_count,
                                   negative=config.negative, sample=config.sample,
                                   hs=config.hs, workers=config.workers)
    if os.path.exists(word2vec_model_path):
        print('{} 模型已存在'.format(word2vec_model_path))
    else:
        logger.info('保存模型 : {}'.format(word2vec_model_path))
        model.save(word2vec_model_path)
    if os.path.exists(word2vec_binary_model_path):
        print('{} 模型已存在'.format(word2vec_binary_model_path))
    else:
        logger.info('保存模型 : {}'.format(word2vec_binary_model_path))
        model.wv.save_word2vec_format(word2vec_binary_model_path, binary=True)


def most_similar(word: str, topk=5):
    """
    获取目标词的相似词
    """
    assert topk > 0, 'topk必须大于0'
    assert os.path.exists(word2vec_model_path) is True, '模型不存在'
    word2vec_model = gensim.models.Word2Vec.load(word2vec_model_path)
    # TODO:如果查找的词不存在，会报错，后续需要处理
    result = word2vec_model.wv.most_similar(word, topn=topk)
    return result


def similarity(word1: str, word2: str):
    """
    比较两个词的相似度
    """
    assert os.path.exists(word2vec_model_path) is True, '模型不存在'
    word2vec_model = gensim.models.Word2Vec.load(word2vec_model_path)
    result = word2vec_model.wv.similarity(word1, word2)
    return result


def get_vector(words: list, binary=False):
    """
    获取list中每个词的词向量
    """
    if not binary:
        assert os.path.exists(word2vec_model_path) is True, '模型不存在'
        word2vec_model = gensim.models.Word2Vec.load(word2vec_model_path)
        word_vectors = {word: word2vec_model.wv[word] for word in words}
    else:
        assert os.path.exists(word2vec_binary_model_path) is True, '模型不存在'
        word2vec_model = KeyedVectors.load_word2vec_format(word2vec_binary_model_path, binary=binary)
        word_vectors = {word: word2vec_model.get_vector(word) for word in words}
    return word_vectors
