# -*- coding:utf-8 -*-
"""
@Project ：zero2one 
@File ：insertion_sort.py
@Author ：cheng
@Date ：2021/1/14
@Description :
    插入排序：将数列分成两部分，第一个数为left部分(下面程序中j)，其他的数为right部分(下面程序中i)，然后遍历right部分中的数，
            插入到left部分中合适位置，遍历结束，left部分就是一个有序数列。
    说明：这里考虑过遍历内层循环时，每次都需要交换相邻两个数，利用切片交换位置的方法，但时间反而增加了，可能时列表切片耗时了，有待进一步研究。
    复杂度： 最优时间复杂度 O(n), 最坏时间复杂度 O(n^2)
    稳定性： 稳定
"""


def insert_sort(list):
    """插入排序"""
    n = len(list)
    for i in range(1, n):
        # 从第i个元素开始向前比较，如果小于前面一个元素，则交换位置
        for j in range(i, 0, -1):
            if list[j] < list[j-1]:
                list[j], list[j-1] = list[j-1], list[j]
            else:
                break


# 测试
if __name__ == '__main__':
    print("=====排序前=========")
    test_list = [3, 6, 1, 8, 5, -20, 100, 50, 200, -32, 123]
    print(test_list)
    print("=====排序后=========")
    insert_sort(test_list)
    print(test_list)
