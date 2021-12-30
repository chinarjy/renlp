# -*- coding = utf-8 -*-
# @Time :2021/10/27 11:21
# @Author:ren.jieye
# @Describe:
# @File : dataset.py
# @Software: PyCharm IT
import torch
from torch.utils.data import Dataset
from RENlp_Intent.utils.utils import load_pkl


class CustomDataset(Dataset):
    def __init__(self, file_path, obj_name):
        self.file = load_pkl(file_path, obj_name)

    def __getitem__(self, item):
        sample = self.file[item]  # 采样
        return sample

    def __len__(self):
        return len(self.file)


def collate_fn(batch):
    batch.sort(key=lambda data: len(data[0]), reverse=True)  # 按照句子长短排序
    lens = [len(data[0]) for data in batch]  # batch中每个句子的长度（由长到短排序）
    max_len = max(lens)  # 取批次里最长的数据，不够长的填充，防止截断造成语义损失

    sent_list = []
    tag_list = []  # y

    # 填充函数
    def _padding(x, max_len):
        return x + [0] * (max_len - len(x))

    for data in batch:
        sent, tag = data
        sent_list.append(_padding(sent, max_len))  # 长度是max_len，不够后面补上0
        tag_list.append(tag)

    return torch.tensor(sent_list), torch.tensor(tag_list)
