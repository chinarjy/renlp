# -*- coding = utf-8 -*-
# @Time :2021/10/27 11:21
# @Author:ren.jieye
# @Describe:
# @File : dataset.py
# @Software: PyCharm IT
import torch
from torch.utils.data import Dataset
from RENlp_NRE.my_nre.utils import load_pkl


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
    head_pos_list = []
    tail_pos_list = []
    mask_pos_list = []
    relation_list = []  # y

    # 填充函数
    def _padding(x, max_len):
        return x + [0] * (max_len - len(x))

    for data in batch:
        sent, head_pos, tail_pos, mask_pos, relation = data
        sent_list.append(_padding(sent, max_len))  # 长度是max_len，不够后面补上0
        head_pos_list.append(_padding(head_pos, max_len))
        tail_pos_list.append(_padding(tail_pos, max_len))
        mask_pos_list.append(_padding(mask_pos, max_len))
        relation_list.append(relation)

    return torch.tensor(sent_list), torch.tensor(head_pos_list), torch.tensor(tail_pos_list), torch.tensor(
        mask_pos_list), torch.tensor(relation_list)
