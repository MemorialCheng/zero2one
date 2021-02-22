# -*- coding:utf-8 -*-
"""
@Project ：zero2one 
@File ：leetcode70_dp.py
@Author ：cheng
@Date ：2021/2/22
@Description : 题目：70-爬楼梯
    假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
    每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
    注意：给定 n 是一个正整数。
"""


class Solution:
    def climbStairs(self, n: int) -> int:
        if n < 3:
            return n
        # 初始化
        dp = [0] * (n + 1)  # +1是因为考虑0阶台阶
        dp[0] = 0
        dp[1] = 1
        dp[2] = 2
        # 推导
        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]
