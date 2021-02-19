# -*- Coding: UTF-8 -*-
"""
@Project ：zero2one
@File ：selection_sort.py
@Author ：cheng
@Date ：2021/1/11
@Description :
    冒泡排序：比较相邻两个数字大小，将比较大（比较小）的那个数交换到后面，不断的交换下去就可以将最大（最小）的那个数放在队列的尾部，
            然后重头再次交换，直到将数列排成有序数列。
    说明：从后往前改变（最大/最小放后面），换言之，先锁定尾部为有序数列。
    复杂度： 最优时间复杂度 O(n), 最坏时间复杂度 O(n^2)
    稳定性： 稳定
"""


def bubble_sort(list):
    """
    冒泡排序
    直接改变原数列
    :param list:
    :return:
    """
    n = len(list)
    if n <= 1:
        return
    for i in range(0, n - 1):
        is_sorted = True
        for j in range(0, n - i - 1):
            if list[j] > list[j + 1]:
                list[j], list[j + 1] = list[j + 1], list[j]
                is_sorted = False
        if is_sorted:
            return


if __name__ == '__main__':
    print("=====排序前=========")
    test_list = [3, 6, 1, 8, 5, -20, 100, 50, 200, -32, 123]
    print(test_list)
    print("=====排序后=========")
    bubble_sort(test_list)
    print(test_list)
