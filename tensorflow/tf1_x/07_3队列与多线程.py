# -*- coding:utf-8 -*-
"""
@Project ：zero2one 
@File ：07_3队列与多线程.py
@Author ：cheng
@Date ：2021/1/21
@Description : 队列与多线程
    修改队列状态的操作主要有Enqueue,EnqueueMany,Dequeue
"""
import tensorflow as tf


def create_queue():
    """
    创建一个队列，并操作里面的元素
    :return:
    """
    # 创建一个先进先出队列，指定队列中最多可以保存两个元素，并指定元素类型为整数。
    q = tf.FIFOQueue(2, "int32")
    # 使用enqueue_many函数初始化队列中的元素，和变量初始化类似，在使用队列之前必须先调用初始化过程。
    init_q = q.enqueue_many([[0, 10], ])
    # 使用dequeue函数将队列中第一个元素弹出队列。
    x = q.dequeue()

    y = x + 1

    # 将y加入队列
    q_inc = q.enqueue([y])

    with tf.Session() as sess:
        # 运行初始化队列的操作
        init_q.run()
        for _ in range(5):
            v, _ = sess.run([x, q_inc])
            print(v)


if __name__ == '__main__':
    create_queue()
