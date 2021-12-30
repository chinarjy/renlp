# -*- coding = utf-8 -*-
# @Time :2021/10/27 11:22
# @Author:ren.jieye
# @Describe:模型训练
# @File : application.py
# @Software: PyCharm IT

# 系统相关
import argparse
import os
# 框架相关
import torch
from torch.utils.data import DataLoader
import torch.optim as optim  # 优化器
import torch.nn as nn
# 自定义
from RENlp_NRE.my_nre.config import config
from RENlp_NRE.my_nre.utils import make_seed, load_pkl
from RENlp_NRE.my_nre.process import process
from RENlp_NRE.my_nre.dataset import CustomDataset, collate_fn
from RENlp_NRE.my_nre import models
from RENlp_NRE.my_nre.trainer import train, validate

# 训练模型列表
__Models__ = {
    "CNN": models.CNN,
    "BiLSTM": models.BiLSTM,
    "BiLSTMPro": models.BiLSTMPro
}

parser = argparse.ArgumentParser(description='关系抽取')  # 创建解析器
parser.add_argument('--model_name', type=str, default=config.model_name, help='model name')  # 添加参数
args = parser.parse_args()  # 解析参数

model_name = args.model_name if args.model_name else config.model_name

make_seed(config.seed)  # 固定值，每次测试结果都是一样的

# 计算设备配置
if config.use_gpu and torch.cuda.is_available():  # 如果选择使用GPU并且有GPU可用
    device = torch.device('cuda', config.gpu_id)
else:
    device = torch.device('cpu')  # 使用CPU
# print(torch.cuda.is_available())

# 数据预处理
if not os.path.exists(config.out_path):  # 如果没有保存好的预处理数据
    process(config.data_path, config.out_path, file_type='csv')

vocab_path = os.path.join(config.out_path, 'vocab.pkl')
train_data_path = os.path.join(config.out_path, 'train.pkl')
test_data_path = os.path.join(config.out_path, 'test.pkl')

# 加载字典
vocab = load_pkl(vocab_path, 'vocab')
vocab_size = len(vocab.word2idx)  # 字典大小

train_dataset = CustomDataset(train_data_path, 'train data')  # type:CustomDataset(DataSet)
test_dataset = CustomDataset(test_data_path, 'test data')

# 创建DataLoader数据加载器
train_dataloader = DataLoader(
    dataset=train_dataset,  # 加载的数据
    batch_size=config.batch_size,  # 每个采样批次的大小
    shuffle=True,  # 打乱
    drop_last=True,  # 取不尽的数据是否舍弃
    collate_fn=collate_fn  # 数据处理函数（填充数据并封装成tensor）
)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn
)

# 动态加载模型
model = __Models__[model_name](vocab_size, config)
model.to(device)
print(model)

# 优化器:保存当前状态并根据计算出的梯度更新参数。
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)  # lr：学习速率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=config.decay_rate,
                                                 patience=config.decay_patience)  # 调度器
# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 测试结果
best_macro_f1, best_macro_epoch = 0, 1
best_micro_f1, best_micro_epoch = 0, 1
best_macro_model, best_micro_model = '', ''

print("******************开始训练*********************")
for epoch in range(1, config.epoch + 1):
    train(epoch, device, train_dataloader, model, optimizer, loss_fn, config)  # 模型训练
    macro_f1, micro_f1 = validate(test_dataloader, model, device, config)  # 模型校验
    model_name = model.save(epoch=epoch)  # 模型保存
    scheduler.step(macro_f1)

    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        best_macro_epoch = epoch
        best_macro_model = model_name
    if micro_f1 > best_micro_f1:
        best_micro_f1 = micro_f1
        best_micro_epoch = epoch
        best_micro_model = model_name

print("*****************模型训练完成*********************")
print(f'best macro f1:{best_macro_f1:.4f}', f'in epoch:{best_macro_epoch}, saved in : {best_macro_model}')
print(f'best micro f1:{best_micro_f1:.4f}', f'in epoch:{best_micro_epoch}, saved in : {best_micro_model}')
