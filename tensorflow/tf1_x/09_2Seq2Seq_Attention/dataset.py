# -*- coding:utf-8 -*-
"""
@Project ：zero2one 
@File ：Dataset.py
@Author ：cheng
@Date ：2021/1/22
@Description : 数据预处理-生成词汇表
"""
from collections import Counter
import json

DATA_TYPE = "chinese"  # 将DATA_TYPE先后设置为chinese,english得到中英文VOCAB文件
if DATA_TYPE == "chinese":  # 翻译语料的中文部分
    RAW_DATA = "./Dataset/en-zh/train.txt.zh"
    WORD_VOCAB = "Vocabulary/zh.vocab"
    IDS_DATA = "./Dataset/en-zh/train_zh_ids.txt"
    VOCAB_SIZE = 4000  # 中文词汇表单词个数
elif DATA_TYPE == "english":  # 翻译语料的英文部分
    RAW_DATA = "./Dataset/en-zh/train.txt.en"
    WORD_VOCAB = "Vocabulary/en.vocab"
    IDS_DATA = "./Dataset/en-zh/train_en_ids.txt"
    VOCAB_SIZE = 40000  # 英文词汇表单词个数


# 词数据集，生成的样本元素为词（对应另外一个类CharDataset为字符）
class WordDataset(object):
    def __init__(self):
        self.word_vocab = {}
        self.len_vocab = 0  # 词索引字典长度

    def build_word_cat_vocab(self):
        """
        生成词汇表
        :return: 将训练集中的词汇保存在词汇表中，一行一个词。
        """
        # 统计词频
        word_list = []
        with open(RAW_DATA, "r", encoding="utf-8") as f:
            for line in f.readlines():
                word_list.extend(line.strip().split())
        # 对词频排序
        word_count = Counter(word_list)
        sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        # 一般情况需要去除低频词，因为PTB数据集已经将低频次替换成<unk>了，所有这里不需要这一步。
        # 将词取出保存
        sorted_words = [item[0] for item in sorted_word_count]

        # 将句子结束符加入到词汇表
        sorted_words = ["<unk>", "<sos>", "<eos>"] + sorted_words

        # 去低频词
        if len(sorted_words) > VOCAB_SIZE:
            sorted_words = sorted_words[:VOCAB_SIZE]

        word_vocab = dict(zip(sorted_words, range(len(sorted_words))))
        self._write_vocab(word_vocab)

    def _write_vocab(self, word_vocab):
        """
        将词索引字典保存为json文件
        :param word_vocab:
        :return:
        """
        with open(WORD_VOCAB, 'w', encoding="utf-8") as fww:
            json.dump(word_vocab, fww, ensure_ascii=False)

    def convert_word_id(self):
        """
        将句子转换为数值列表
        """
        self._read_vocab()
        word_ids_list = []  # 保存转换为数字的句子列表
        with open(RAW_DATA, "r", encoding="utf-8") as f:
            for sentence in f.readlines():
                word_list = sentence.strip().split() + ["<eos>"]
                # 将每个单词替换为词汇表中的编号
                out_line = [str(self.word_vocab[word]) if word in self.word_vocab.keys()
                            else str(self.word_vocab["<unk>"]) for word in word_list]
                word_ids_list.append(" ".join(out_line) + "\n")
        # 将转换为数字编号的列表保存起来
        with open(IDS_DATA, "w", encoding="utf-8") as wf:
            wf.writelines(word_ids_list)

    def _read_vocab(self):
        with open(WORD_VOCAB, "r") as wi:
            self.word_vocab = json.load(wi)
            self.len_vocab = len(self.word_vocab)  # 保存词索引字典的长度


if __name__ == '__main__':
    all_data = WordDataset()
    # all_data.build_word_cat_vocab()
    all_data.convert_word_id()
