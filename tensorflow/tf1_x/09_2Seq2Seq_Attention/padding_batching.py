# -*- coding:utf-8 -*-
"""
@Project ：zero2one 
@File ：padding_batching.py
@Author ：cheng
@Date ：2021/2/1
@Description : padding and batching
"""
import tensorflow as tf

MAX_LEN = 50                        # 限定句子的最大单词数量
SOS_ID = 1                          # 目标语言词汇表中<sos>的ID


# 使用Dataset从一个文件中读取一个语言的数据。
# 数据的格式为每一句话，单词已经转化为单词编号
def MakeDataset(file_path):
    """
    数据batching,产生最后输入数据格式
    :param file_path: 数据路径
    :return: dataset-　每个句子－对应的长度组成的TextLineDataset类的数据集对应的张量
    """
    dataset = tf.data.TextLineDataset(file_path)

    # map(function, sequence[, sequence, ...]) -> list
    # 通过定义可以看到，这个函数的第一个参数是一个函数，剩下的参数是一个或多个序列，返回值是一个集合。
    # function可以理解为是一个一对一或多对一函数，map的作用是以参数序列中的每一个元素调用function函数，返回包含每次function函数返回值的list。
    # lambda argument_list: expression
    # 其中lambda是Python预留的关键字,argument_list和expression由用户自定义
    # argument_list参数列表, expression 为函数表达式
    # 根据空格将单词编号切分开并放入一个一维向量
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    # 将字符串形式的单词编号转化为整数
    dataset = dataset.map(lambda string: tf.string_to_number(string, tf.int32))
    # 统计每个句子的单词数量，并与句子内容一起放入Dataset
    dataset = dataset.map(lambda x: (x, tf.size(x)))
    return dataset


def MakeSrcTrgDataset(src_path, trg_path, batch_size):
    """
    从源语言文件src_path和目标语言文件trg_path中分别读取数据，并进行填充和batching操作
    :param src_path: 源语言，即被翻译的语言,英语.
    :param trg_path: 目标语言，翻译之后的语言,汉语.
    :param batch_size: batch的大小
    :return:
    """
    # 首先分别读取源语言数据和目标语言数据
    src_data = MakeDataset(src_path)
    trg_data = MakeDataset(trg_path)
    # 通过zip操作将两个Dataset合并为一个Dataset，现在每个Dataset中每一项数据ds由4个张量组成
    # ds[0][0]是源句子
    # ds[0][1]是源句子长度
    # ds[1][0]是目标句子
    # ds[1][1]是目标句子长度
    # https://blog.csdn.net/qq_32458499/article/details/78856530这篇博客看一下可以细致了解一下Dataset这个库，以及.map和.zip的用法
    dataset = tf.data.Dataset.zip((src_data, trg_data))

    # 删除内容为空(只包含<eos>)的句子和长度过长的句子
    def FilterLength(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        # tf.logical_and 相当于集合中的and做法，后面两个都为true最终结果才会为true，否则为false
        # tf.greater Returns the truth value of (x > y),所以以下所说的是句子长度必须得大于一也就是不能为空的句子
        # tf.less_equal Returns the truth value of (x <= y),所以所说的是长度要小于最长长度
        src_len_ok = tf.logical_and(tf.greater(src_len, 1), tf.less_equal(src_len, MAX_LEN))
        trg_len_ok = tf.logical_and(tf.greater(trg_len, 1), tf.less_equal(trg_len, MAX_LEN))
        return tf.logical_and(src_len_ok, trg_len_ok) # 两个都满足才返回true

    # filter接收一个函数Func并将该函数作用于dataset的每个元素，根据返回值True或False保留或丢弃该元素，True保留该元素，False丢弃该元素
    # 最后得到的就是去掉空句子和过长的句子的数据集
    dataset = dataset.filter(FilterLength)

    # 解码器需要两种格式的目标句子：
    # 1.解码器的输入(trg_input), 形式如同'<sos> X Y Z'
    # 2.解码器的目标输出(trg_label), 形式如同'X Y Z <eos>'
    # 上面从文件中读到的目标句子是'X Y Z <eos>'的形式，我们需要从中生成'<sos> X Y Z'形式并加入到Dataset
    # 编码器只有输入,没有输出,而解码器有输入也有输出，输入为<sos>＋(除去最后一位eos的label列表)
    # 例如train.en最后都为2,ｉｄ为２就是eos
    def MakeTrgInput(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        # tf.concat用法 https://blog.csdn.net/qq_33431368/article/details/79429295
        trg_input = tf.concat([[SOS_ID], trg_label[:-1]], axis=0)
        return ((src_input, src_len), (trg_input, trg_label, trg_len))
    dataset = dataset.map(MakeTrgInput)

    # 随机打乱训练数据
    dataset = dataset.shuffle(10000)

    # 规定填充后的输出的数据维度
    padded_shapes = (
        (tf.TensorShape([None]),    # 源句子是长度未知的向量
         tf.TensorShape([])),       # 源句子长度是单个数字
        (tf.TensorShape([None]),    # 目标句子(解码器输入)是长度未知的向量
         tf.TensorShape([None]),    # 目标句子(解码器目标输出)是长度未知的向量
         tf.TensorShape([]))        # 目标句子长度(输出)是单个数字
    )
    # 调用padded_batch方法进行batching操作
    batched_dataset = dataset.padded_batch(batch_size, padded_shapes)

    return batched_dataset


if __name__ == '__main__':
    IDS_DATA_EN = "./Dataset/en-zh/train_en_ids.txt"
    IDS_DATA_ZH = "./Dataset/en-zh/train_zh_ids.txt"
    MakeSrcTrgDataset(IDS_DATA_EN, IDS_DATA_ZH, 20)

