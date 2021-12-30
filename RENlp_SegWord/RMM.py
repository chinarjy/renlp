# -*- coding = utf-8 -*-
# @Time :2021/9/24 8:41
# @Author:ren.jieye
# @Describe:规则分词-逆向最大匹配法
# @File : RMM.py
# @Software: PyCharm IT
from load_segword_dict import load2dict

class RMM(object):
    def __init__(self):
        self.window_size = 4

    def cut(self, sentence):
        result = []
        index = len(sentence)
        dic = ['研究', '研究生', '生命', '命', '的', '起源', '南京', '南京市', '市长', '长江', '长江大桥', '江', '大桥']
        dic = load2dict()
        while index > 0:
            for size in range(index - self.window_size, index):
                piece = sentence[size:index]
                if piece in dic:
                    index = size + 1
                    break
            index = index - 1
            result.append(piece)
        result.reverse()
        return result
