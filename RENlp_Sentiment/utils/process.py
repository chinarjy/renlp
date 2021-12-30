# -*- coding = utf-8 -*-
# @Time :2021/10/27 11:21
# @Author:ren.jieye
# @Describe:数据预处理
# @File : process.py
# @Software: PyCharm IT
# 系统相关
import os
import codecs
import csv
import json
# 自定义
from RENlp_Sentiment.utils.utils import load_csv, load_json, ensure_dir, save_pkl, load_txt
from RENlp_Sentiment.config import config
from RENlp_Sentiment.utils.vocab import Vocab

# 第三方
import jieba


# 判断数据集中是否存在关系字段
def exist_relation(file, file_type):
    with codecs.open(file, 'r', encoding='utf-8') as f:
        if file_type == 'csv':
            f = csv.DictReader(f)
        for line in f:
            keys = list(line.keys())
            try:
                num = keys.index('relation')
            except:
                num = -1
            return int(num)


# 分词
def split_sentences(raw_data):
    jieba.load_userdict('data/一汽大众领域词典.txt')  # 定制自己的领域词典，用以提升分词效果
    new_data_list = []
    for data in raw_data:
        new_sent = jieba.lcut(data)
        new_data_list.append([[data], new_sent])
    return new_data_list


# 构建词典
def bulid_vocab(raw_datas, out_path):
    if config.word_segment:
        vocab = Vocab('word')
        for data in raw_datas:
            vocab.add_sentences(data[1])  # data[-2]：切好词的句子
    else:
        vocab = Vocab('char')
        for data in raw_datas:
            vocab.add_sentences(data[0])  # data[0]：原始没切词的句子
    vocab.trim(config.min_freq)

    ensure_dir(out_path)

    vocab_path = os.path.join(out_path, 'vocab.pkl')
    vocab_txt = os.path.join(out_path, 'vocab.txt')
    save_pkl(vocab_path, vocab, 'vocab')

    with codecs.open(vocab_txt, 'w', encoding='utf-8') as f:
        f.write(os.linesep.join([word for word in vocab.word2idx.keys()]))
    return vocab, vocab_path


# 获取位置编码特征
def get_pos_feature(sent_len, entity_pos, entity_len, pos_limit):
    """
    :param sent_len:句子长度
    :param entity_pos:编码
    :param entity_len:长度
    :param pos_limit:
    :return:
    """

    left = list(range(-entity_pos, 0))  # 左侧编码
    middle = [0] * entity_len  # 中间编码
    right = list(range(1, sent_len - entity_pos - entity_len + 1))  # 右侧编码
    pos = left + middle + right  # 位置编码信息
    # 保证位置编码在【-50,50】之间
    for i, p in enumerate(pos):
        if p > pos_limit:
            pos[i] = pos_limit
        if p < -pos_limit:
            pos[i] = -pos_limit
    # 都变成正值，放入神经网络比较好计算
    pos = [p + pos_limit + 1 for p in pos]
    return pos


def get_mask_feature(entities_pos, sen_len):
    """
    获取mask编码
    :param entities_pos:
    :param sen_len:
    :return:
    """
    left = [1] * (entities_pos[0] + 1)
    middle = [2] * (entities_pos[1] - entities_pos[0] - 1)
    right = [3] * (sen_len - entities_pos[1])
    return left + middle + right


def bulid_data(raw_data, vocab):
    sents = []

    if vocab.name == 'word':
        for data in raw_data:
            # 文本特征（用的是每个词的ID)
            sent = [vocab.word2idx.get(w, 1) for w in data[1]]
            sents.append(sent)
    else:
        for data in raw_data:
            sent = [vocab.word2idx.get(w, 1) for w in data[0]]
            sents.append(sent)
    return sents


# 把数据集中的relation从汉字转换成编码数字
def tag_tokenize(tags, file):
    """
    :param tags:
    :param file:
    :return:
    """
    tags_list = []
    tags_dict = {}
    out = []
    with codecs.open(file, encoding='utf-8') as f:
        for line in f:
            tags_list.append(line.strip())
    for i, rel in enumerate(tags_list):
        tags_dict[rel] = i
    for rel in tags:
        out.append(tags_dict[rel.strip()])
    return out


def has_tag(file_path):
    with codecs.open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            nums = len(line.split('***'))
            if nums == 2:
                continue
            else:
                print('训练数据标注错误-错误内容: {}'.format(line))
                f.close()
                return False
    f.close()
    return True


def process(data_path, out_path, file_type):
    print('*************************开始数据预处理*************************')
    # file_type = file_type.lower()  # 小写
    # assert file_type in ['csv', 'json']  # 判断原始数据类型是否是csv或json

    print('*************************加载原始数据*************************')

    train_fp_path = os.path.join(data_path, 'train.txt')
    test_fp_path = os.path.join(data_path, 'test.txt')
    tag_fp_path = os.path.join(data_path, 'tag.txt')

    # 判断是否所有数据都进行了标记
    train_data_right = has_tag(train_fp_path)
    test_data_right = has_tag(test_fp_path)

    train_data_list = []
    train_tag_list = []
    if train_data_right and test_data_right:  # 数据标注没问题
        train_datas, train_tags = load_txt(train_fp_path)
        test_datas, test_tags = load_txt(test_fp_path)

        if config.is_CN and config.word_segment:  # 中文分词
            train_datas = split_sentences(train_datas)
            test_datas = split_sentences(test_datas)

        print("*****************构建词典******************")
        vocab, vocab_path = bulid_vocab(train_datas, out_path)

        print("******************构建train模型数据******************")
        train_sents = bulid_data(train_datas, vocab)

        print("******************构建test模型数据******************")
        test_sents = bulid_data(test_datas, vocab)

        print("******************构建关系数据******************")
        # 关系进行编号，并将训练集和测试集中的关系转换成编号
        train_tag_token = tag_tokenize(train_tags, tag_fp_path)
        test_tag_token = tag_tokenize(test_tags, tag_fp_path)

        #  处理完的数据保存到硬盘上
        ensure_dir(out_path)
        train_data = list(
            zip(train_sents, train_tag_token)
        )
        test_data = list(
            zip(test_sents, test_tag_token)
        )

        train_data_path = os.path.join(out_path, 'train.pkl')
        test_data_path = os.path.join(out_path, 'test.pkl')

        save_pkl(train_data_path, train_data, 'train data')
        save_pkl(test_data_path, test_data, 'test data')

        print("******************数据预处理完成******************")
