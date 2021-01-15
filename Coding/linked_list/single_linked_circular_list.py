# -*- coding:utf-8 -*-
"""
@Project ：zero2one 
@File ：single_linked_circular_list.py
@Author ：cheng
@Date ：2021/1/15
@Description : 单向循环链表
https://blog.csdn.net/hfutdog/article/details/94374147?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522159505107019195265951765%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=159505107019195265951765&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v3~pc_rank_v2-1-94374147.first_rank_ecpm_v3_pc_rank_v2&utm_term=python%E4%B8%AD%E5%8D%95%E5%90%91%E5%BE%AA%E7%8E%AF%E9%93%BE%E8%A1%A8
"""



class Node(object):
    """创建结点类"""

    def __init__(self, item):
        self.item = item
        self.next = None


class SingleCycleLinkList(object):
    """单向循环链表"""

    def __init__(self, node=None):
        """初始化列表"""
        if node is None:
            self.__head = node
        else:
            self.__head = Node(node)
            node.next = node

    def is_empty(self):
        """判断链表是否为空"""
        return self.__head is None

    def length(self):
        """统计链表长度"""
        if self.is_empty():
            return 0
        cur = self.__head  # cur表示游标
        count = 1  # count游标个数
        while cur.next is not self.__head:
            count += 1
            cur = cur.next
        return count

    def travel(self):
        """遍历整个链表"""
        if self.is_empty():
            return
        cur = self.__head
        while cur.next is not self.__head:
            print(cur.item, end=" ")
            cur = cur.next
        print(cur.item)

    def add(self, item):
        """链表头部添加元素,头插法"""
        node = Node(item)
        if self.is_empty():  # 当链表为空时
            self.__head = node
            node.next = node
        else:  # 链表不为空时
            cur = self.__head
            while cur.next is not self.__head:
                cur = cur.next  # 该循环为了使cur指向尾结点
            node.next = self.__head
            self.__head = node
            cur.next = node

    def append(self, item):
        """链表尾部添加元素  尾插法"""
        node = Node(item)
        if self.is_empty():  # 当链表为空时
            self.__head = node
            node.next = node
        else:  # 链表不为空时
            cur = self.__head
            while cur.next is not self.__head:
                cur = cur.next
            # 退出循环的时候，cur指向的尾结点
            cur.next = node
            node.next = self.__head

    def insert(self, pos, item):
        """在指定位置(下标)添加结点, 与单链表一样"""
        if pos <= 0:
            self.add(item)
        elif pos >= self.length():
            self.append(item)
        else:
            cur = self.__head
            count = 0
            while count < (pos - 1):
                count += 1
                cur = cur.next
            node = Node(item)
            node.next = cur.next
            cur.next = node

    def search(self, item):
        """查找结点是否存在item这个值"""
        if self.is_empty():
            return False
        cur = self.__head
        while cur.next is not self.__head:
            if cur.item == item:
                return True
            else:
                cur = cur.next
        if cur.item == item:
            return True
        return False

    def remove(self, item):
        """删除结点，按值删除"""
        if self.is_empty():
            return
        cur = self.__head
        pre = None
        while cur.next is not self.__head:
            if cur.item == item:   # 找到了要删除的元素
                if cur == self.__head:  # 在头部找到了元素
                    tail = self.__head  # tail用来指向尾结点
                    while tail.next is not self.__head:
                        tail = tail.next
                    tail.next = cur.next
                    self.__head = cur.next
                else:  # 在中间找到
                    pre.next = cur.next
                return
            else:  # 不是要找的元素，移动游标
                pre = cur
                cur = cur.next
        #  退出循环后，cur指向最后一个结点
        if cur.item == item:
            if cur is self.__head:  # 链表只有一个结点时
                self.__head = None
            else:
                pre.next = cur.next


if __name__ == '__main__':
    singCycleLinkList = SingleCycleLinkList()
    print(singCycleLinkList.is_empty())
    print(singCycleLinkList.length())
    print("=================")
    singCycleLinkList.append(20)
    print(singCycleLinkList.is_empty())
    print(singCycleLinkList.length())
    print("====================")
    singCycleLinkList.add(20)
    singCycleLinkList.add(30)
    singCycleLinkList.add(40)
    singCycleLinkList.append(50)
    singCycleLinkList.travel()
    print("============")
    singCycleLinkList.remove(40)
    singCycleLinkList.travel()
