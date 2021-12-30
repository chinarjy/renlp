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
            data = line.replace('\r\n', '').split('***')[0]
            tag = line.replace('\r\n', '').split('***')[1]
            data_list.append(data)
            tag_list.append(tag)
    train_fp.close()
    return data_list, tag_list


# 保存词典
def sava_pkl(path, obj, obj_name):
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
    矩阵式LED
    动态可变
    自动解锁
    AR-Hud
    高清高亮
    12吋
    ID.Light
    智能语音
    远程空调
    远程充电
    30色
    环抱式
    温馨舒适
    触控开关
    科技感
    震动
    最大扭矩
    非常快
    静谧性
    舒适
    节约能耗
    一举两得
    自动驾驶
    方便
    安全
    制动
    距离
    支撑性
    稳
    发飘
    低速轻
    转弯半径
    清晰
    小道
    30万
    None
    :param key:
    :return:
    """
    relations = {
        '0': '矩阵式LED',
        '1': '动态可变',
        '2': '自动解锁',
        '3': 'AR-Hud',
        '4': '高清高亮',
        '5': '12吋',
        '6': 'ID.Light',
        '7': '智能语音',
        '8': '远程空调',
        '9': '远程充电',
        '10': '30色',
        '11': '环抱式',
        '12': '温馨舒适',
        '13': '触控开关',
        '14': '科技感',
        '15': '震动',
        '16': '最大扭矩',
        '17': '非常快',
        '18': '静谧性',
        '19': '舒适',
        '20': '节约能耗',
        '21': '一举两得',
        '22': '自动驾驶',
        '23': '方便',
        '24': '安全',
        '25': '制动',
        '26': '距离',
        '27': '支撑性',
        '28': '稳',
        '29': '发飘',
        '30': '低速轻',
        '31': '转弯半径',
        '32': '清晰',
        '33': '小道',
        '34': '30万',
        '35': 'None'
    }
    result = {
        'tag': relations.get(str(key))
    }
    return result
