# -*- coding:utf-8 -*-
"""
@Project ：zero2one 
@File ：train_model.py
@Author ：cheng
@Date ：2021/2/1
@Description : 训练模型
"""
import tensorflow as tf
from padding_batching import MakeSrcTrgDataset

SRC_TRAIN_DATA = './Dataset/en-zh/train_en_ids.txt'  # 源语言输入文件
TRG_TRAIN_DATA = './Dataset/en-zh/train_zh_ids.txt'  # 目标语言输入文件
CHECKPOINT_PATH = './Trained_model/seq2seq_ckpt'  # checkpoint保存路径
HIDDEN_SIZE = 1024                  # LSTM的隐藏层规模
NUM_LAYERS = 2                      # 深层循环神经网络中LSTM结构的层数
SRC_VOCAB_SIZE = 40000              # 源语言词汇表大小
TRG_VOCAB_SIZE = 4000               # 目标语言词汇表大小
BATCH_SIZE = 100                    # 训练数据batch的大小
NUM_EPOCH = 5                       # 使用训练数据的轮数
KEEP_PROB = 0.8                     # 节点不被dropout的概率
MAX_GRAD_NORM = 5                   # 用于控制梯度膨胀的梯度大小上限
SHARE_EMB_AND_SOFTMAX = True        # 在softmax层和词向量层之间共享参数


class NMTModel(object):
    """
    function: 模型初始化
    """
    def __init__(self):

        # 定义编码器和解码器所使用的LSTM结构
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])
        # 为源语言和目标语言分别定义词向量
        self.src_embedding = tf.get_variable('src_emb', [SRC_VOCAB_SIZE, HIDDEN_SIZE])
        self.trg_embedding = tf.get_variable('trg_emb', [TRG_VOCAB_SIZE, HIDDEN_SIZE])
        # 定义softmax层的变量
        if SHARE_EMB_AND_SOFTMAX:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable('weight', [HIDDEN_SIZE, TRG_VOCAB_SIZE])
        self.softmax_bias = tf.get_variable('softmax_loss', [TRG_VOCAB_SIZE])

    def forward(self, src_input, src_size, trg_input, trg_label, trg_size):
        """
        在forward函数中定义模型的前向计算图
        :param src_input: 编码器输入（源数据）
        :param src_size: 输入大小
        :param trg_input: 解码器输入（目标数据）
        :param trg_label: 解码器输出（目标数据）
        :param trg_size: 输出大小
        :return:
        """
        batch_size = tf.shape(src_input)[0]
        # 将输入和输出单词转为词向量（rnn中输入数据都要转换成词向量）
        # 相当于input中的每个id对应的embedding中的向量转换
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)
        trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)
        # 在词向量上进行dropout
        src_emb = tf.nn.dropout(src_emb, KEEP_PROB)
        trg_emb = tf.nn.dropout(trg_emb, KEEP_PROB)
        # 使用dynamic_rnn构造编码器
        # 编码器读取源句子每个位置的词向量，输出最后一步的隐藏状态enc_state
        # 因为编码器是一个双层LSTM，因此enc_state是一个包含两个LSTMStateTuple类的tuple，
        # 每个LSTMStateTuple对应编码器中一层的状态
        # enc_outputs是顶层LSTM在每一步的输出，它的维度是[batch_size, max_time, HIDDEN_SIZE]
        # seq2seq模型中不需要用到enc_outputs,而attention模型会用到它
        with tf.variable_scope('encoder'):
            enc_outputs, enc_state = tf.nn.dynamic_rnn(self.enc_cell, src_emb, src_size, dtype=tf.float32)
        # 使用dynamic_rnn构造解码器
        # 解码器读取目标句子每个位置的词向量，输出的dec_outputs为每一步顶层LSTM的输出
        # dec_outputs的维度是[batch_size, max_time, HIDDEN_SIZE]
        # initial_state=enc_state表示用编码器的输出来初始化第一步的隐藏状态
        # 编码器最后编码结束最后的状态为解码器初始化的状态
        with tf.variable_scope('decoder'):
            dec_outputs, _ = tf.nn.dynamic_rnn(self.dec_cell, trg_emb, trg_size, initial_state=enc_state)
        # 计算解码器每一步的log perplexity
        # 输出重新转换成shape为[,HIDDEN_SIZE]
        output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
        #  计算解码器每一步的softmax概率值
        logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
        #  交叉熵损失函数，算loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(trg_label, [-1]), logits=logits)
        # 在计算平均损失时，需要将填充位置的权重设置为0，以避免无效位置的预测干扰模型的训练
        label_weights = tf.sequence_mask(trg_size, maxlen=tf.shape(trg_label)[1], dtype=tf.float32)
        label_weights = tf.reshape(label_weights, [-1])
        cost = tf.reduce_sum(loss * label_weights)
        cost_per_token = cost / tf.reduce_sum(label_weights)
        # 定义反向传播操作
        trainable_variables = tf.trainable_variables()
        # 控制梯度大小，定义优化方法和训练步骤
        # 算出每个需要更新的值的梯度，并对其进行控制
        grads = tf.gradients(cost / tf.to_float(batch_size), trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
        # 利用梯度下降优化算法进行优化.学习率为1.0
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        # 相当于minimize的第二步，正常来讲所得到的list[grads,vars]由compute_gradients得到，返回的是执行对应变量的更新梯度操作的op
        train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
        return cost_per_token, train_op


def run_epoch(session, cost_op, train_op, saver, step):
    """
    使用给定的模型model上训练一个epoch，并返回全局步数，每训练200步便保存一个checkpoint
    :param session: 会话
    :param cost_op: 计算loss的操作op
    :param train_op: 训练的操作op
    :param saver: 保存model的类
    :param step: 训练步数
    :return:
    """
    # 训练一个epoch
    # 重复训练步骤直至遍历完Dataset中所有数据
    while True:
        try:
            # 运行train_op并计算cost_op的结果也就是损失值，训练数据在main()函数中以Dataset方式提供
            cost, _ = session.run([cost_op, train_op])
            # 步数为１０的倍数进行打印
            if step % 10 == 0:
                print('After %d steps, per token cost is %.3f' % (step, cost))
            # 每200步保存一个checkpoint
            if step % 200 == 0:
                saver.save(session, CHECKPOINT_PATH, global_step=step)
            step += 1
        except tf.errors.OutOfRangeError:
            break
    return step


def main():
    # 定义初始化函数
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    # 定义训练用的循环神经网络模型
    with tf.variable_scope('nmt_model', reuse=None, initializer=initializer):
        train_model = NMTModel()
    # 定义输入数据
    data = MakeSrcTrgDataset(SRC_TRAIN_DATA, TRG_TRAIN_DATA, BATCH_SIZE)
    iterator = data.make_initializable_iterator()
    (src, src_size), (trg_input, trg_label, trg_size) = iterator.get_next()
    # 定义前向计算图，输入数据以张量形式提供给forward函数
    cost_op, train_op = train_model.forward(src, src_size, trg_input, trg_label, trg_size)
    # 训练模型
    # 保存模型
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
        # 初始化全部变量
        tf.global_variables_initializer().run()
        # 进行NUM_EPOCH轮数
        for i in range(NUM_EPOCH):
            print('In iteration: %d' % (i + 1))
            sess.run(iterator.initializer)
            step = run_epoch(sess, cost_op, train_op, saver, step)


if __name__ == '__main__':
    main()
