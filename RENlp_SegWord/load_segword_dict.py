# -*- coding = utf-8 -*-
# @Time :2021/9/24 9:30
# @Author:ren.jieye
# @Describe:中文分词词典下载
# @File : load_segword_dict.py
# @Software: PyCharm IT
import os


def load2dict():
    file_path = 'data/中文分词词库.txt'
    words = [line.strip() for line in open(file_path, 'r', encoding='utf-8').readlines()]
    return words


class LoadSegWord2Dict:
    pass


if __name__ == '__main__':
    load2dict()
