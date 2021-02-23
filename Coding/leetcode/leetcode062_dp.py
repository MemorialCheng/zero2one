# -*- coding:utf-8 -*-
"""
@Project ：zero2one 
@File ：leetcode062_dp.py
@Author ：cheng
@Date ：2021/2/23
@Description : 题目：62. 不同路径
一个机器人位于一个 m x n网格的左上角 （起始点在下图中标记为 “Start” ）。
机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
问总共有多少条不同的路径？

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/unique-paths
"""


class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        if m <= 0 or n <= 0:
            return 0
        # 初始化
        dp = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)]

        # 推导
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[m - 1][n - 1]
