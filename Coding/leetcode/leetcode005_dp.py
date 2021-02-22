# -*- coding:utf-8 -*-
"""
@Project ：zero2one 
@File ：leetcode005_dp.py
@Author ：cheng
@Date ：2021/2/22
@Description : 题目5：最长回文子串
    给你一个字符串 s，找到 s 中最长的回文子串。
"""


class Solution:
    def longestPalindrome(self, s: str) -> str:
        size = len(s)
        if size < 2:
            return s

        dp = [[False]*size for _ in range(size)]

        max_len = 1
        start = 0
        # 对角线元素先填为True
        for i in range(size):
            dp[i][i] = True
        # 推导，因为要求i<=j,所以填对角线上面元素
        for j in range(1, size):
            for i in range(0, j):
                if s[i] == s[j]:
                    if j - i < 3:  # 边界判断：子串不构成区间，即j-1-(i+1)+1<2
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i + 1][j - 1]
                else:  # 首尾字符不相等时直接填False
                    dp[i][j] = False
                # 计算回文子串开始下标和最大长度
                if dp[i][j] and j-i+1 > max_len:
                    max_len = j-i+1
                    start = i
        return s[start:start + max_len]
