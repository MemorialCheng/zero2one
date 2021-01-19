# -*- coding:utf-8 -*-
"""
@Project ：zero2one 
@File ：LeNet5_predictor.py
@Author ：cheng
@Date ：2021/1/19
@Description : MNIST手写数字识别
    LeNet-5实现, 用测试集进行数字识别预测, 预测类别
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import LeNet5_inference
import LeNet5_train
import numpy as np


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None,
                                        LeNet5_inference.IMAGE_SIZE,
                                        LeNet5_inference.IMAGE_SIZE,
                                        LeNet5_inference.NUM_CHANNELS], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, LeNet5_inference.OUTPUT_NODE], name="y-input")

        # 测试数据
        xs = mnist.test.images  # 10000*763
        reshaped_xs = np.reshape(xs, [-1,
                                      LeNet5_inference.IMAGE_SIZE,
                                      LeNet5_inference.IMAGE_SIZE,
                                      LeNet5_inference.NUM_CHANNELS])

        test_feed = {x: reshaped_xs, y_: mnist.test.labels}

        y = LeNet5_inference.inference(x, None)

        # 预测类别
        predicted_num = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(LeNet5_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(LeNet5_train.MODEL_SAVE_PATH)
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
    tf.app.run()

