# -*- coding = utf-8 -*-
# @Time :2021/10/27 13:34
# @Author:ren.jieye
# @Describe:
# @File : BasicModule.py
# @Software: PyCharm IT
import torch
import torch.nn as nn
import time
from RENlp_Sentiment.utils.utils import ensure_dir


class BasicModule(nn.Module):
    '''
    封装nn.Module,提供save和load方法
    '''

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = 'BasicModule'

    '''
    加载指定路径的模型
    '''

    def load(self, path):
        self.load_state_dict(torch.load(path))

    '''
    保存模型
    '''

    def save(self, epoch=0, name=None):
        prefix = 'model_file/'
        ensure_dir(prefix)
        if name is None:
            name = prefix + self.model_name + '_' + f'epoch{epoch}_'
            name = time.strftime(name + '%m%d_%H_%M_%S.pth')
        else:
            name = prefix + name + '_' + self.model_name + '_' + f'epoch{epoch}_'
            name = time.strftime(name + '%m%d_%H_%M_%S.pth')
        print(name)
        torch.save(self.state_dict(), name)
        return name
