# -*- coding:utf-8 -*-
"""
@Project ：zero2one 
@File ：05_2变量管理.py
@Author ：cheng
@Date ：2021/1/13
@Description : TensorFlow通过变量名称获取变量的机制主要通过
        tf.get_variable和tf.variable_scope函数实现。
        说明：tf.get_variable函数可以用来创建或者获取变量，当用于创建变量是，与tf.Variable的功能基本等价。
            不同之处是tf.Variable函数的name参数是可选的，tf.get_variable是必填的。
"""
import tensorflow as tf


def demo1():
    with tf.variable_scope("foo"):
        v = tf.get_variable("v", [1], initializer=tf.constant_initializer(1.0))

    # reuse=True表示这个上下文管理器内的tf.get_variable会直接获取已经创建的变量
    with tf.variable_scope("foo", reuse=True):
        v1 = tf.get_variable("v", [1])
        print(v == v1)  # 输出True,表示v,v1代表的是相同的tensorflow中的变量
