# -*- Coding:utf-8 -*-
"""
@Project ：zero2one 
@File ：03_2张量_会话的使用.py
@Author ：cheng
@Date ：2021/1/12
@Description : 张量(ensor)
    零阶张量表示标量(scalar)，也就是一个数，常量；
    一阶张量表示向量(vector)，也就是一个一维数组；
    n阶张量可以理解为一个n维数组。
"""

import tensorflow as tf

a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")

result = tf.add(a, b, name="add")
print(result)  # Tensor("add:0", shape=(2,), dtype=float32)

# 会话的使用
with tf.Session() as sess:
    # print(sess.run(result))  # 与下面这一句有相同功能
    print(result.eval())

