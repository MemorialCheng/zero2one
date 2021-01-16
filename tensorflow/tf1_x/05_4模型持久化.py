# -*- coding:utf-8 -*-
"""
@Project ：zero2one 
@File ：05_4模型持久化.py
@Author ：cheng
@Date ：2021/1/14
@Description : tf.train.Saver.save(sess, save_path)会保存三个文件：
        1. model.ckpt.meta,它保存了TensorFlow计算图的结构；
        2. model.ckpt,这个文件保存了TensorFlow每个变量的取值；
        3. checkpoint,这个文件保存了一个目录下所有的模型文件列表。
"""
import tensorflow as tf


def save_model_demo():
    """
    保存模型
    :return:
    """
    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
    result = v1 + v2

    # 声明tf.train.Saver()类用于保存模型
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(result.eval())
        saver.save(sess, "./tem/model/model.ckpt")


def get_model_demo():
    """
    重新定义图上运算的方式来加载模型
    :return:
    """
    # 使用和保存模型代码中一样的方式来声明变量
    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
    result = v1 * v2

    # 声明tf.train.Saver()类用于保存模型
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, "./tem/model/model.ckpt")
        print(sess.run(result))


def get_model_demo1():
    """
    直接加载模型
    :return:
    """
    saver = tf.train.import_meta_graph("./tem/model/model.ckpt.meta")

    with tf.Session() as sess:
        saver.restore(sess, "./tem/model/model.ckpt")
        # 通过张量的名称来获取张量
        print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))


if __name__ == '__main__':
    # save_model_demo()
    get_model_demo()
    # get_model_demo1()

