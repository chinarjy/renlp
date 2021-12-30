# -*- coding = utf-8 -*-
# @Time :2021/9/24 9:03
# @Author:ren.jieye
# @Describe:规则分词-双向最大匹配法
# @File : BMM.py
# @Software: PyCharm IT
from MM import MM
from RMM import RMM


class BMM(object):
    def __init__(self):
        self.window_size = 4

    @staticmethod
    def cut(sentence):
        tokenizer_mm = MM()
        tokenizer_rmm = RMM()
        result_mm = tokenizer_mm.cut(sentence)
        result_rmm = tokenizer_rmm.cut(sentence)
        if len(result_mm) > len(result_rmm):
            return result_rmm
        elif len(result_mm) < len(result_rmm):
            return result_mm
        else:
            one_word_mm = 0
            one_word_rmm = 0
            for word in result_mm:
                if len(word) == 1:
                    one_word_mm = one_word_mm + 1
            for word in result_rmm:
                if len(word) == 1:
                    one_word_rmm = one_word_rmm + 1
            if one_word_mm > one_word_rmm:
                return result_rmm
            else:
                return result_mm
