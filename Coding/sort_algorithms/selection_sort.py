# -*- Coding: UTF-8 -*-
"""
@Project ：zero2one 
@File ：selection_sort.py
@Author ：cheng
@Date ：2021/1/11
@Description :
    选择排序：每一轮循环选择剩余数列中最大（最小）的那个数下标，与头部待有序的那个数交换位置。
    说明：从前往后改变（最大/最小放前面），换言之，先锁定头部为有序数列。
    复杂度： 最优、最坏时间复杂度都是 O(n^2)
    稳定性： 不稳定（当升序每次选择最大值时不稳定），但一般升序我们选择最小值。
"""
import timeit

def select_sort(list):
    """
    传统的，利用两次循环遍历
    :param list:
    :return:
    """
    n = len(list)
    if n <= 1:
        return list
    for i in range(0, n-1):
        min_index = i
        for j in range(i + 1, n):
            if list[min_index] > list[j]:
                min_index = j
        list[min_index], list[i] = list[i], list[min_index]


def select_sort_op(list):
    """
    较传统方式，利用python自带函数min和index求解，执行时间更快
    :param list:
    :return:
    """
    n = len(list)
    if n <= 1:
        return list
    for i in range(0, n-1):
        if list[i] != min(list[i:]):
            min_index = list.index(min(list[i:]))
            list[i], list[min_index] = list[min_index], list[i]


# 测试
if __name__ == '__main__':
    print("=====排序前=========")
    test_list = [3, 6, 1, 8, 5, -20, 100, 50, 200, -32, 123]
    print(test_list)
    print("=====排序后=========")
    select_sort(test_list)
    print(test_list)
    print(timeit.timeit("select_sort(test_list)", "from __main__ import select_sort,test_list", number=100))

# 传统：0.0012285269986023195
# 优化：0.0005667589939548634
