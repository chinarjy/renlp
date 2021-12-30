# -*- coding = utf-8 -*-
# @Time :2021/10/27 11:21
# @Author:ren.jieye
# @Describe:
# @File : trainer.py
# @Software: PyCharm IT
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def train(epoch, device, data_loader, model, optimizer, loss_fn, config):
    """
    :param epoch: 训练轮次
    :param device: 设备（CPU/GPU)
    :param data_loader: 数据加载器
    :param model: 模型
    :param optimizer: 优化器
    :param loss: 损失函数
    :param cofig: 参数配置
    :return:
    """
    model.train() # 开始训练
    total_loss = []

    for batch_idx, batch in enumerate(data_loader):  # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        *x, y = [data.to(device) for data in batch]  # 每个特征的tensor向量
        optimizer.zero_grad()  # 优化器梯度清零
        y_pred = model(x)
        loss = loss_fn(y_pred, y)  # 损失函数
        loss.backward()  # 反向传播
        optimizer.step()  # 参数优化
        total_loss.append(loss)

        # 记录日志
        data_cal = len(data_loader.dataset) if batch_idx == len(data_loader) else batch_idx * len(y)

        if (config.train_log and batch_idx % config.log_interval == 0) or batch_idx == len(data_loader):
            print('Traing epoch:{}[{}/{}({:.0f}%)]\tLoss:{:.6f}'.format(epoch,
                                                                        data_cal,
                                                                        len(data_loader.dataset),
                                                                        100. * batch_idx / len(data_loader),
                                                                        loss.item()
                                                                        ))


def validate(data_loader, model, device, config):
    """
    :param data_loader:
    :param model:
    :param device:
    :param config:
    :return:
    """
    model.eval()

    with torch.no_grad():
        total_y_true = np.empty(0)
        total_y_pred = np.empty(0)
        for batch_idx, batch in enumerate(data_loader):
            *x, y = [data.to(device) for data in batch]
            y_pred = model(x)

            y_pred = y_pred.argmax(dim=-1)

            try:
                y, y_pred = y.numpy(), y_pred.numpy()
            except:
                y, y_pred = y.cpu().numpy(), y_pred.cpu().numpy()

            total_y_true = np.append(total_y_true, y)
            total_y_pred = np.append(total_y_pred, y_pred)

        total_f1 = []

        for average in config.f1_norm:
            p, r, f1, _ = precision_recall_fscore_support(total_y_true, total_y_pred, average=average)

            print(f'{average} metrics:[p：{p:.4f}, r:{r:.4f}, f1:{f1:.4f}]')

            total_f1.append(f1)

    return total_f1
