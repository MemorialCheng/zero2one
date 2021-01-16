# -*- coding:utf-8 -*-
"""
@Project ：zero2one 
@File ：05_3利用变量管理改进MNIST手写数字识别.py
@Author ：cheng
@Date ：2021/1/13
@Description : 
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 1.设置输入和输出节点的个数, 配置神经网络的参数
INPUT_NODE = 784  # 输入节点
OUTPUT_NODE = 10  # 输出节点
LAYER1_NODE = 500  # 隐藏层数

BATCH_SIZE = 100  # 每次batch打包的样本个数

# 模型相关的参数
LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARAZTION_RATE = 0.0001  # 正则化项在损失函数中的系数
TRAINING_STEPS = 5000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


def inference(input_tensor, reuse=False):
    with tf.variable_scope("layer1", reuse=reuse):
        weights = tf.get_variable("weights", [INPUT_NODE, LAYER1_NODE], initializer=tf.truncated_normal_initializer(0.1))
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.truncated_normal_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope("layer2", reuse=reuse):
        weights = tf.get_variable("weights", [LAYER1_NODE, OUTPUT_NODE], initializer=tf.truncated_normal_initializer(0.1))
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.truncated_normal_initializer(0.0))
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)
    return layer2


# 3.定义训练过程。
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    # 计算不含滑动平均类的前向传播结果
    y = inference(x)



