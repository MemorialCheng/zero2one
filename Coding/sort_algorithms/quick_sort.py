# -*- coding:utf-8 -*-
"""
@Project ：zero2one 
@File ：quick_sort.py
@Author ：cheng
@Date ：2021/1/15
@Description : 快速排序
    复杂度： 最优时间复杂度 O(nlogn), 最坏时间复杂度 O(n^2)
    稳定性： 不稳定
    O(nlogn)解释说明：每一趟排序执行n遍，需要拆分2^x = n ==> x=logn次，所以总的 nlogn
"""


def quick_sort(list, first, last):
    """
    传统方法：指针移动的思想
    能很好的诠释快速排序思想，但繁琐。
    直接改变原数列。
    """
    if first >= last:  # 终止条件
        return
    mid_value = list[first]
    low = first
    high = last
    while low < high:
        # high 左移
        while low < high and list[high] >= mid_value:
            high -= 1
        list[low] = list[high]
        # low 右移
        while low < high and list[low] <= mid_value:
            low += 1
        list[high] = list[low]
    # 退出循环时，low == high
    list[low] = mid_value

    # 将列表分为两部分分别排序（first, low-1)和（low+1, last)
    quick_sort(list, first, low-1)
    quick_sort(list, low+1, last)


def quick_sort_op(iList):
    """
    方法巧妙，充分利用Python，值得学习。
    返回有序数列，未直接改变原数列
    :param iList:
    :return:
    """
    if len(iList) <= 1:
        return iList
    left = []
    right = []
    for i in iList[1:]:
        if i <= iList[0]:
            left.append(i)
        else:
            right.append(i)
    return quick_sort_op(left) + [iList[0]] + quick_sort_op(right)


if __name__ == '__main__':
    # print("=====排序前=========")
    # test_list = [3, 6, 1, 8, 5, -20, 100, 50, 200, -32, 123]
    # print(test_list)
    # print("=====排序后=========")
    # quick_sort(test_list, 0, len(test_list)-1)
    # print(test_list)

    print("=====排序前=========")
    test_list = [3, 6, 1, 8, 5, -20, 100, 50, 200, -32, 123]
    print(test_list)
    print("=====排序后=========")
    print(quick_sort_op(test_list))
