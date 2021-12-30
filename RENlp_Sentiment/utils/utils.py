# -*- coding = utf-8 -*-
# @Time :2021/10/27 11:21
# @Author:ren.jieye
# @Describe:
# @File : utils.py
# @Software: PyCharm IT

# 系统相关
import random
import codecs
import json
import csv
import os
# 框架相关
import torch
# 第三方
import numpy as np
import pickle as pkl

"""
    确保路径存在，
"""


def ensure_dir(directory):
    """
    判断目录是否存在，不存在就创建
    :param path:
    :return:
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def make_seed(num):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    torch.cuda.manual_seed_all(num)


# 加载csv文件
def load_csv(file):
    data_list = []
    with codecs.open(file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for line in reader:
            data = list(line.values())
            data_list.append(data)
    return data_list


# 加载json文件
def load_json(file):
    data_list = []
    with codecs.open(file, encoding='utf-8') as f:
        for line in f:
            json_data = json.load(line)
            data = list(json_data.values())
            data_list.append(data)
    return data_list


def load_txt(file_path):
    data_list = []
    tag_list = []
    # 加载训练数据
    with codecs.open(file_path, 'r', encoding='utf-8') as train_fp:
        lines = train_fp.readlines()
        for line in lines:
            data = line.strip().split('***')[0]
            tag = line.strip().split('***')[1]
            data_list.append(data)
            tag_list.append(tag)
    train_fp.close()
    return data_list, tag_list


# 保存词典
def save_pkl(path, obj, obj_name):
    print(f'save {obj_name} in {path}')
    with codecs.open(path, 'wb') as f:
        pkl.dump(obj, f)


# 加载数据
def load_pkl(path, obj_name):
    print(f'load {obj_name} in {path}')
    with codecs.open(path, 'rb') as f:
        data = pkl.load(f)
    return data


#  结果返回
def get_result(key):
    """
    负向
    中性
    正向
    :param key:
    :return:
    """
    relations = {
        '0': '正向',
        '1': '中性',
        '2': '负向'
    }
    result = {
        'tag': relations.get(str(key))
    }
    return result


#  结果返回
def get_result_eat(key):
    """
    negative
    positive
    :param key:
    :return:
    """
    relations = {
        '0': 'negative',
        '1': 'positive'
    }
    result = {
        'tag': relations.get(str(key))
    }
    return result
