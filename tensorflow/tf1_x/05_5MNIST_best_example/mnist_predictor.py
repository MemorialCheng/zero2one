# -*- coding:utf-8 -*-
"""
@Project ：zero2one 
@File ：mnist_eval.py
@Author ：cheng
@Date ：2021/1/15
@Description : MNIST最佳实战样例
    用测试集进行数字识别预测, 预测类别
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import mnist_train


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        y = mnist_inference.inference(x, None)

        # 预测类别
        predicted_num = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            # tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录下最新模型的文件名
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                predicted_num_score = sess.run(predicted_num, feed_dict=test_feed)
                print("After %s training step(s), test predicted_num_score =\n %r" % (
                    global_step, predicted_num_score[:100]))  # 只打印了前100个预测值
            else:
                print('No checkpoint file found')
                return


def main(argv=None):
    mnist = input_data.read_data_sets("../datasets/MNIST_data", one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    main()
    # After 30000 training step(s), test predicted_num_score =
    #  array([7, 2, 1, 0, 4, 1, 4, 9, 5, 9, 0, 6, 9, 0, 1, 5, 9, 7, 3, 4, 9, 6,
    #        6, 5, 4, 0, 7, 4, 0, 1, 3, 1, 3, 4, 7, 2, 7, 1, 2, 1, 1, 7, 4, 2,
    #        3, 5, 1, 2, 4, 4, 6, 3, 5, 5, 6, 0, 4, 1, 9, 5, 7, 8, 9, 3, 7, 4,
    #        6, 4, 3, 0, 7, 0, 2, 9, 1, 7, 3, 2, 9, 7, 7, 6, 2, 7, 8, 4, 7, 3,
    #        6, 1, 3, 6, 9, 3, 1, 4, 1, 7, 6, 9])

