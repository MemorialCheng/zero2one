# -*- coding:utf-8 -*-
"""
@Project ：zero2one 
@File ：03_3TensorFlow变量.py
@Author ：cheng
@Date ：2021/1/12
@Description :
    变量与张量什么关系？
    变量的声明函数tf.Variable是一个运算。这个运算的输出结果就是一个张量。
"""

import tensorflow as tf


def forward_pro_1():
    """
    用常量进行简单的前向传播
    :return:
    """
    w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))

    x = tf.constant([[0.7, 0.9]])

    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)

    with tf.Session() as sess:
        # sess.run(w1.initializer)  # 初始化w1
        # sess.run(w2.initializer)  # 初始化w2
        sess.run(tf.global_variables_initializer())

        print(sess.run(y))  # 输出[[3.957578]]


def forward_pro_2():
    """
    利用placeholder实现前向传播算法
    :return:
    """
    w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

    x = tf.placeholder(tf.float32, shape=(1, 2), name="input")

    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        print(sess.run(y, feed_dict={x: [[0.7, 0.9]]}))


def loss_example(y, y_):
    """
    一个简单的损失函数
    :param y: 预测值
    :param y_: 真实值
    :return:
    """
    # sigmoid函数将y转换为0~1之间的数值，转换后y代表预测是正样本的概率，1-y代表预测是负样本的概率。
    y = tf.sigmoid(y)
    # 定义损失函数（交叉熵损失函数）
    cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
                                    + (1-y) * tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))
    # 定义学习率
    learning_rate = 0.001
    # 定义反向传播算法来优化神经网络的参数
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)


if __name__ == '__main__':
    forward_pro_1()
    forward_pro_2()
