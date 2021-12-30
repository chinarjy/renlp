#!/usr/bin/env python
# encoding: utf-8
"""
@author: jieye.ren
@contact: chinarjy@163.com
@software: pycharm
@file: test.py
@time: 2021/12/2 10:21 AM
@desc: 日志
"""
import logging
import os

log_path = 'data/log'
if not os.path.exists(log_path):
    os.makedirs(log_path)

logger = logging.getLogger("client_log")

# Log等级总开关
logger.setLevel(logging.INFO)

# 创建handler，用于输出到控制台、写入日志文件
stream_handler = logging.StreamHandler()
log_file_handler = logging.FileHandler(filename=os.path.join(log_path, "training_log.log"), encoding="utf-8")

# 定义handler的输出格式
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")

stream_handler.setFormatter(formatter)
log_file_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logger.addHandler(log_file_handler)
