# -*- coding:utf-8 -*-
"""
@Project ：zero2one 
@File ：LeNet5_eval.py
@Author ：cheng
@Date ：2021/1/19
@Description : MNIST手写数字识别
    LeNet-5实现, 用验证集评估准确率
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

        # 验证数据
        xs = mnist.validation.images
        reshaped_xs = np.reshape(xs, (-1,
                                      LeNet5_inference.IMAGE_SIZE,
                                      LeNet5_inference.IMAGE_SIZE,
                                      LeNet5_inference.NUM_CHANNELS))

        validate_feed = {x: reshaped_xs, y_: mnist.validation.labels}

        y = LeNet5_inference.inference(x, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(LeNet5_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(LeNet5_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
            else:
                print('No checkpoint file found')
                return


def main(argv=None):
    mnist = input_data.read_data_sets("../datasets/MNIST_data", one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()

# After 30000 training step(s), validation accuracy = 0.9914
