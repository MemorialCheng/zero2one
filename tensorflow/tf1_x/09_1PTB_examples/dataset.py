# -*- coding:utf-8 -*-
"""
@Project ：zero2one 
@File ：dataset.py
@Author ：cheng
@Date ：2021/1/22
@Description : 数据预处理-生成词汇表
"""
from collections import Counter
import json


RAW_TRAIN_DATA = "Dataset/ptb.train.txt"  # 原始的训练数据文件
RAW_VALID_DATA = "Dataset/ptb.valid.txt"  # 原始的验证集数据文件
RAW_TEST_DATA = "Dataset/ptb.test.txt"  # 原始的测试集数据文件

TRAIN_IDS = "Dataset/ptb_train_ids.txt"  # 将单词替换为单词编号后的输出文件
VALID_IDS = "Dataset/ptb_valid_ids.txt"  # 将单词替换为单词编号后的输出文件
TEST_IDS = "Dataset/ptb_test_ids.txt"  # 将单词替换为单词编号后的输出文件

WORD_VOCAB = "Vocabulary/ptb_word_to_id.json"  # 词汇表文件-根据训练集生成


# 词数据集，生成的样本元素为词（对应另外一个类CharDataset 为字符）
class WordDataset(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.text_data_frame = None

        self.word_vocab = {}
        self.len_vocab = 0  # 词索引字典长度

    def build_word_cat_vocab(self):
        """
        生成词汇表
        :return: 将训练集中的词汇保存在词汇表中，一行一个词。
        """
        # 统计词频
        word_list = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                word_list.extend(line.strip().split())
        # 对词频排序
        word_count = Counter(word_list)
        sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        # 一般情况需要去除低频词，因为PTB数据集已经将低频次替换成<unk>了，所有这里不需要这一步。
        # 将词取出保存
        sorted_words = [item[0] for item in sorted_word_count]

        # 将句子结束符加入到词汇表
        sorted_words = ["<eos>"] + sorted_words

        word_vocab = dict(zip(sorted_words, range(len(sorted_words))))
        self._write_vocab(word_vocab)

    def _write_vocab(self, word_vocab):
        """
        将词索引字典保存为json文件
        :param word_vocab:
        :return:
        """
        with open(WORD_VOCAB, 'w') as fww:
            json.dump(word_vocab, fww)

    def convert_word_id(self, save_path):
        """
        将词列表转换为数值列表
        """
        self._read_vocab()
        word_ids_list = []  # 保存转换为数字的句子列表
        with open(self.file_path, "r", encoding="utf-8") as f:
            for sentence in f.readlines():
                word_list = sentence.strip().split() + ["<eos>"]
                # 将每个单词替换为词汇表中的编号
                out_line = [str(self.word_vocab[word]) if word in self.word_vocab.keys()
                            else str(self.word_vocab["<unk>"]) for word in word_list]
                word_ids_list.append(" ".join(out_line) + "\n")
        # 将转换为数字编号的列表保存起来
        with open(save_path, "w", encoding="utf-8") as wf:
            wf.writelines(word_ids_list)

    def _read_vocab(self):
        with open(WORD_VOCAB, "r") as wi:
            self.word_vocab = json.load(wi)
            self.len_vocab = len(self.word_vocab)  # 保存词索引字典的长度


if __name__ == '__main__':
    all_data = WordDataset(RAW_VALID_DATA)
    # all_data.build_word_cat_vocab()
    all_data.convert_word_id(VALID_IDS)



