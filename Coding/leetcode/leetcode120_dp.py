# -*- coding:utf-8 -*-
"""
@Project ：zero2one 
@File ：leetcode120_dp.py
@Author ：cheng
@Date ：2021/2/23
@Description : 题目：120. 三角形最小路径和。 （与64题类似）
给定一个三角形 triangle ，找出自顶向下的最小路径和。
每一步只能移动到下一行中相邻的结点上。相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。
也就是说，如果正位于当前行的下标 i ，那么下一步可以移动到下一行的下标 i 或 i + 1 。

示例 1：
输入：triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
输出：11
解释：如下面简图所示：
   2
  3 4
 6 5 7
4 1 8 3
自顶向下的最小路径和为11（即，2 + 3 + 5 + 1 = 11）。
来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/triangle
"""


class Solution:
    def minimumTotal(self, triangle: list[list[int]]) -> int:
        if not triangle or not triangle[0]:
            return 0
        m = len(triangle)  # 三角形行数
        if m == 1:
            return triangle[0][0]

        # 初始化
        # dp[i][j]表示到达i行j列的最小路径和
        # 理解为填n*n方格的左下部分三角形
        dp = [[0] * m for _ in range(m)]
        dp[0][0] = triangle[0][0]

        # 推导
        for i in range(1, m):  # 行
            dp[i][0] = dp[i - 1][0] + triangle[i][0]  # 每行第一个元素
            dp[i][i] = dp[i - 1][i - 1] + triangle[i][i]  # 对角线
            for j in range(1, i):  # 列
                dp[i][j] = min(dp[i - 1][j], dp[i - 1][j - 1]) + triangle[i][j]
        return min(dp[m - 1])

