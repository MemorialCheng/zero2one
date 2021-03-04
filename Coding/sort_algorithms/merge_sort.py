# -*- coding:utf-8 -*-
"""
@Project ：zero2one 
@File ：merge_sort.py
@Author ：cheng
@Date ：2021/1/15
@Description : 归并排序
    复杂度：最优时间复杂度 O(nlogn),最坏时间复杂度 O(nlogn)
    稳性性： 稳定
"""


def merge_sort(list):
    """
    归并排序
    返回有序数列，未直接改变原数列
    """
    n = len(list)
    if n < 2:
        return list
    mid = n // 2

    left_list = merge_sort(list[:mid])  # left_list 采用归并排序后形成的有序的新列表
    right_list = merge_sort(list[mid:])  # right_list 采用归并排序后形成的有序的新列表

    # 将两个有序的子序列合并为一个新的整体
    # left_point, right_point分别表示两个子序列指向第一个元素的下标
    left_point, right_point = 0, 0
    # result 用来保存合并后的列表
    result = []

    while left_point < len(left_list) and right_point < len(right_list):
        if left_list[left_point] <= right_list[right_point]:
            result.append(left_list[left_point])
            left_point += 1
        else:
            result.append(right_list[right_point])
            right_point += 1
    # 退出循环时，left_list和right_list两个子序列中，其中一个已经全部保存到result列表中，
    # 现需将另一个子序列剩下的全部元素直接保存到result列表中
    result += left_list[left_point:]
    result += right_list[right_point:]
    return result


# 测试
if __name__ == '__main__':
    print("=====排序前=========")
    test_list = [3, 6, 1, 8, 5, -20, 100, 50, 200, -32, 123]
    print(test_list)
    print("=====排序后=========")
    print(merge_sort(test_list))
