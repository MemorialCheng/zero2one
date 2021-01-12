# -*- Coding:utf-8 -*-
"""
@Project ：zero2one 
@File ：03_1计算图的使用.py
@Author ：cheng
@Date ：2021/1/11
@Description : 
"""
import tensorflow as tf
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def graph_example():
    # tf.reset_default_graph()
    g1 = tf.Graph()
    with g1.as_default():
        # 在计算图g1中定义变量“v”,并设置初始值为0
        v = tf.compat.v1.get_variable("v", shape=[1], initializer=tf.zeros_initializer)

    g2 = tf.Graph()
    g2 = tf.compat.v1.Graph()
    with g2.as_default():
        # 在计算图g1中定义变量“v”,并设置初始值为0
        v = tf.compat.v1.get_variable("v", shape=[1], initializer=tf.ones_initializer)

    # 读取变量
    with tf.compat.v1.Session(graph=g1) as sess:
        tf.compat.v1.global_variables_initializer().run()
        with tf.compat.v1.variable_scope("", reuse=True):
            print(sess.run(tf.compat.v1.get_variable("v")))  # [0.]

    with tf.compat.v1.Session(graph=g2) as sess:
        tf.compat.v1.global_variables_initializer().run()
        with tf.compat.v1.variable_scope("", reuse=True):
            print(sess.run(tf.compat.v1.get_variable("v")))  # [1.]

    # with tf.Session(graph=g2) as sess:
    #     tf.global_variables_initializer().run()
    #     with tf.variable_scope("", reuse=True):
    #         print(sess.run(tf.get_variable("v")))


if __name__ == '__main__':
    graph_example()
