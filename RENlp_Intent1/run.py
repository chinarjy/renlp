#!/usr/bin/env python
# encoding: utf-8
"""
@author: jieye.ren
@contact: chinarjy@163.com
@software: pycharm
@file: run.py
@time: 2021/12/3 11:04 AM
@desc: 机器学习方法进行意图识别
"""
from models.LogisticModel import LogisticModel
from models.SVCModel import SVCModel

if __name__ == "__main__":
    # 实例化对象
    model = LogisticModel()
    # model = SVCModel()
    model.train()
