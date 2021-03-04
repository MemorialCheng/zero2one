# -*- coding:utf-8 -*-
"""
@Project ：zero2one 
@File ：shell_sort.py
@Author ：cheng
@Date ：2021/1/14
@Description :
    希尔排序:核心是将序列分为几个子序列，对子序列进行插入排序。
            list[j] < list[j-1] 升序 操作的是前面有序数据
    复杂度： 最优时间复杂度根据序列步长策略有关, 最坏时间复杂度 O(n^2)
    稳定性： 不稳定
"""


def shell_sort(list):
    n = len(list)
    if n < 2:
        return
    gap = n // 2  # 定义步长
    # gap为1,2,......
    while gap > 0:
        for i in range(gap, n):
            # 与普通插入排序区别就在于gap
            # j = [i, i-gap, i-gap-gap, ..., 0]
            for j in range(i, 0, -gap):
                if list[j] < list[j-gap]:
                    list[j], list[j-gap] = list[j-gap], list[j]
                else:
                    break
        # 缩短步长，减半
        gap //= 2


if __name__ == '__main__':
    print("=====排序前=========")
    test_list = [3, 6, 1, 8, 5, -20, 100, 50, 200, -32, 123]
    print(test_list)
    print("=====排序后=========")
    shell_sort(test_list)
    print(test_list)
