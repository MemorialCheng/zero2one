# -*- coding:utf-8 -*-
"""
@Project ：zero2one 
@File ：08_1LSTM前向传播.py
@Author ：cheng
@Date ：2021/1/21
@Description : LSTM前向传播过程.(以下程序为结构介绍，不可运行）
    定义LSTM结构，TensorFlow中通过一句简单命令即可实现tf.nn.rnn_cell.BasicLSTMCell。
    该函数具体讲解参考：https://blog.csdn.net/weixin_42713739/article/details/103391813
"""
import tensorflow as tf


def fully_connected(input):
    """
    全连接层
    :param input:
    :return:
    """
    return 0


def inference_lstm():
    """
    LSTM前向传播简单定义
    :return:
    """
    lstm_hidden_size = 10  # 隐状态的个数
    batch_size = 32
    num_steps = 15  # 暂用num_steps表示时间步长
    current_input = tf.constant(0.1, shape=[13, 5])  # 定义输入

    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)
    # state包含h,c两个状态
    state = lstm.zero_state(batch_size=batch_size, dtype=tf.float32)
    loss = 0.0  # 损失函数

    for i in range(num_steps):
        if i > 0:  # 在第一个时刻声明LSTM结构中使用的变量，在之后的时刻需要复用之前定义好的变量。
            tf.get_variable_scope().reuse_variables()

        lstm_output, state = lstm(current_input, state)
        # 将当前时刻lstm结构的输出传入一个全连接层得到最后的输出
        final_output = fully_connected(lstm_output)
        # 计算当前时刻输出的损失
        # loss += calc_loss(final_output, expected_output)


def inference_deeplstm():
    """
    深层循环神经网络
    :return:
    """
    num_of_layers = 3  # 层数
    lstm_hidden_size = 10  # 循环体的隐状态个数
    batch_size = 32
    num_steps = 15  # 暂用num_steps表示时间步长
    current_input = tf.constant(0.1, shape=[13, 5])  # 定义输入

    # 定义循环体
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)
    # 不能使用[lstm_cell]*N的形式来初始化MultiRNNCell，否则TensorFlow会在每一层之间共享参数
    deep_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell for _ in range(num_of_layers)])

    state = deep_lstm.zero_state(batch_size=batch_size, dtype=tf.float32)

    for i in range(num_steps):
        if i > 0:
            tf.get_variable_scope().reuse_variables()

        lstm_output, state = deep_lstm(current_input, state)
        # 将当前时刻lstm结构的输出传入一个全连接层得到最后的输出
        final_output = fully_connected(lstm_output)
        # 计算当前时刻输出的损失
        # loss += calc_loss(final_output, expected_output)
tf.nn.rnn_cell.DropoutWrapper