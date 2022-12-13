# -*- coding: utf-8 -*-
"""
@File : 神经网络进行气温预测.py
@Author : cheng
@Date : 2022/12/13
@Description :
    利用pytorch搭建神经网络，进行气温预测。

数据表中
    year,moth,day,week分别表示的具体的时间
    temp_2：前天的最高温度值
    temp_1：昨天的最高温度值
    average：在历史中，每年这一天的平均最高温度值
    actual：这就是我们的标签值了，当天的真实最高温度
    friend：这一列可能是凑热闹的，你的朋友猜测的可能值，咱们不管它就好了
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import datetime
from sklearn import preprocessing


def handle_date(data_df):
    """
    处理时间数据
    :param data_df:
    :return:
    """
    # 分别得到年，月，日
    years = data_df['year']
    months = data_df['month']
    days = data_df['day']

    # datetime格式
    dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
    dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
    return dates


def chart_view(dates, features):
    # 准备画图
    # 指定默认风格
    plt.style.use('fivethirtyeight')

    # 设置布局
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    fig.autofmt_xdate(rotation=45)

    # 标签值
    ax1.plot(dates, features['actual'])
    ax1.set_xlabel('')
    ax1.set_ylabel('Temperature')
    ax1.set_title('Max Temp')

    # 昨天
    ax2.plot(dates, features['temp_1'])
    ax2.set_xlabel('')
    ax2.set_ylabel('Temperature')
    ax2.set_title('Previous Max Temp')

    # 前天
    ax3.plot(dates, features['temp_2'])
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Temperature')
    ax3.set_title('Two Days Prior Max Temp')

    # 我的逗逼朋友
    ax4.plot(dates, features['friend'])
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Temperature')
    ax4.set_title('Friend Estimate')

    plt.tight_layout(pad=2)
    plt.show()  # 展示图像


def data_pre_processing(features):
    # 独热编码
    features = pd.get_dummies(features)
    # 标签
    labels = np.array(features['actual'])
    # 在特征中去掉标签
    features = features.drop('actual', axis=1)
    # 名字单独保存一下，以备后患
    feature_list = list(features.columns)
    # 转换成合适的格式
    features = np.array(features)
    # 标准化
    input_features = preprocessing.StandardScaler().fit_transform(features)
    return labels, feature_list, input_features


def my_model(labels, input_features):
    x = torch.tensor(input_features, dtype=float)

    y = torch.tensor(labels, dtype=float)

    # 权重参数初始化
    weights = torch.randn((14, 128), dtype=float, requires_grad=True)
    biases = torch.randn(128, dtype=float, requires_grad=True)
    weights2 = torch.randn((128, 1), dtype=float, requires_grad=True)
    biases2 = torch.randn(1, dtype=float, requires_grad=True)

    learning_rate = 0.001
    losses = []

    for i in range(1000):
        # 计算隐层
        hidden = x.mm(weights) + biases
        # 加入激活函数
        hidden = torch.relu(hidden)
        # 预测结果
        predictions = hidden.mm(weights2) + biases2
        # 通计算损失
        loss = torch.mean((predictions - y) ** 2)
        losses.append(loss.data.numpy())

        # 打印损失值
        if i % 100 == 0:
            print('loss:', loss)
        # 返向传播计算
        loss.backward()

        # 更新参数
        weights.data.add_(- learning_rate * weights.grad.data)
        biases.data.add_(- learning_rate * biases.grad.data)
        weights2.data.add_(- learning_rate * weights2.grad.data)
        biases2.data.add_(- learning_rate * biases2.grad.data)

        # 每次迭代都得记得清空
        weights.grad.data.zero_()
        biases.grad.data.zero_()
        weights2.grad.data.zero_()
        biases2.grad.data.zero_()


if __name__ == '__main__':

    features = pd.read_csv('./data/temps.csv')

    datas = handle_date(features)

    # chart_view(datas, features)

    labels, feature_list, input_features = data_pre_processing(features)

    my_model(labels, input_features)