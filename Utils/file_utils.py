#!/usr/bin/env python
# encoding: utf-8
"""
@author: jieye.ren
@contact: chinarjy@163.com
@software: pycharm
@file: file_utils.py
@time: 2021/12/13 1:48 PM
@desc: 文件操作工具类
"""
import os
from Utils import logger


def file_exist(filename):
    if os.path.exists(filename):
        os.remove(filename)
        logger.info('移除了{}文件'.format(filename))