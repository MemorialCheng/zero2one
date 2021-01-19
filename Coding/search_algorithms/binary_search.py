# -*- coding:utf-8 -*-
"""
@Project ：zero2one 
@File ：binary_search.py
@Author ：cheng
@Date ：2021/1/18
@Description : 二分查找
    复杂度：最优时间复杂度 O(1)，最坏时间复杂度 O(logn)
"""


def binary_search_by_recursion(list, item):
    """二分查找，递归法"""
    n = len(list)
    if n > 0:
        mid = n//2
        if list[mid] == item:
            return True
        elif list[mid] > item:
            return binary_search_by_recursion(list[:mid], item)
        else:
            return binary_search_by_recursion(list[mid+1:], item)
    return False


def binary_search_by_no_recursion(list, item):
    """二分查找，非递归法"""
    n = len(list)
    first = 0
    last = n-1
    while first <= last:
        mid = (first + last)//2
        if list[mid] == item:
            return True
        elif list[mid] > item:
            last = mid - 1
        else:
            first = mid + 1
    return False


if __name__ == '__main__':
    test_list = [-32, -20, 1, 3, 6, 8, 50, 100, 123, 200]
    print("=========递归法===========")
    print(binary_search_by_recursion(test_list, 100))
    print(binary_search_by_recursion(test_list, 10))
    print("=========非递归法===========")
    print(binary_search_by_no_recursion(test_list, 100))
    print(binary_search_by_no_recursion(test_list, 10))
