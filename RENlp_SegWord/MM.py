# -*- coding = utf-8 -*-
# @Time :2021/9/24 8:33
# @Author:ren.jieye
# @Describe:规则分词-正向最大匹配法
# @File : MM.py
# @Software: PyCharm IT
from load_segword_dict import load2dict

class MM(object):
    def __init__(self):
        self.window_size = 4

    def cut(self, sentence):
        result = []
        index = 0
        sentence_length = len(sentence)
        dic = ['研究', '研究生', '生命', '命', '的', '起源', '南京', '南京市', '市长', '长江', '长江大桥', '江', '大桥']
        dic = load2dict()
        while sentence_length > index:
            for size in range(self.window_size + index, index, -1):
                piece = sentence[index:size]
                if piece in dic:
                    index = size - 1
                    break
            index = index + 1
            result.append(piece)
        return result
