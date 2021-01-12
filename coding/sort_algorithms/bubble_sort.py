# -*- coding: UTF-8 -*-
"""
@Project ：zero2one
@File ：selection_sort.py
@Author ：cheng
@Date ：2021/1/11
@Description : 冒泡排序
"""


def bubble_sort(list):
    """
    冒泡排序
    :param list:
    :return:
    """
    n = len(list)
    for i in range(0, n-1):
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