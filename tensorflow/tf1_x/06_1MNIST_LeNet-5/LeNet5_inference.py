# -*- coding:utf-8 -*-
"""
@Project ：zero2one 
@File ：LeNet5_inference.py
@Author ：cheng
@Date ：2021/1/16
@Description : MNIST手写数字识别
    LeNet-5实现, 前向传播
"""

import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

CONV1_DEEP = 32
FILTER1_SIZE = 5

CONV2_DEEP = 64
FILTER2_SIZE = 5

FC_SIZE = 512


# 通过tf.get_variable函数来获取变量
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def get_biases_variable(shape, rate):
    biases = tf.get_variable("bias", shape, initializer=tf.constant_initializer(rate))
    return biases


def inference(input_tensor, regularizer, train=False):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = get_weight_variable([FILTER1_SIZE, FILTER1_SIZE, NUM_CHANNELS, CONV1_DEEP], None)
        conv1_biases = get_biases_variable([CONV1_DEEP], 0.0)
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = get_weight_variable([FILTER2_SIZE, FILTER2_SIZE, CONV1_DEEP, CONV2_DEEP], None)
        conv2_biases = get_biases_variable([CONV2_DEEP], 0.0)
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # 因下面要经过全连接层,将7*7*64矩阵拉直成一个向量
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        # -1表示一个batch中数据的个数;这样reshaped表示一个batch的向量
        reshaped = tf.reshape(pool2, [-1, nodes])

    with tf.variable_scope('layer5-fc1'):
        # 只有全连接的权重需要加入正则化
        fc1_weights = get_weight_variable([nodes, FC_SIZE], regularizer)
        fc1_biases = get_biases_variable([FC_SIZE], 0.1)

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        # dropout在训练时会随机将部分节点的输出改为0，dropout可以避免过拟合问题，一般只在全连接层而不是卷积层和池化层使用
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_weights = get_weight_variable([FC_SIZE, NUM_LABELS], regularizer)
        fc2_biases = get_biases_variable([NUM_LABELS], 0.1)
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit
