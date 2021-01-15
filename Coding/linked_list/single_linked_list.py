# -*- coding:utf-8 -*-
"""
@Project ：zero2one 
@File ：single_linked_list.py
@Author ：cheng
@Date ：2021/1/15
@Description : 单链表
https://blog.csdn.net/kang19950919/article/details/88890853?ops_request_misc=&request_id=&biz_id=102&utm_term=python%E4%B8%AD%E9%93%BE%E8%A1%A8&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-88890853
"""


class Node(object):
    """创建结点类"""

    def __init__(self, item):
        self.item = item
        self.next = None


class SingleLinkList(object):
    """单链表"""

    def __init__(self, node=None):
        """初始化列表"""
        if node is None:
            self.__head = node
        else:
            self.__head = Node(node)

    def is_empty(self):
        """判断链表是否为空"""
        return self.__head is None

    def length(self):
        """统计链表长度"""
        cur = self.__head  # cur表示游标
        count = 0  # count游标个数
        while cur is not None:
            count += 1
            cur = cur.next
        return count

    def travel(self):
        """遍历整个链表"""
        cur = self.__head
        while cur is not None:
            print(cur.item, end=" ")
            cur = cur.next

    def add(self, item):
        """链表头部添加元素  头插法"""
        node = Node(item)
        node.next = self.__head
        self.__head = node

    def append(self, item):
        """链表尾部添加元素  尾插法"""
        node = Node(item)
        # 如果链表为空，需要特殊处理
        if self.is_empty():
            self.__head = node
        else:
            cur = self.__head
            while cur.next is not None:
                cur = cur.next
            # 退出循环的时候，cur指向的尾结点
            cur.next = node

    def insert(self, pos, item):
        """
        按位置插入
        :param pos: 位置下标
        :param item: 值
        :return:
        """
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
        cur = self.__head
        while cur is not None:
            if cur.item == item:
                return True
            cur = cur.next
        return False

    def remove(self, item):
        """删除结点，按值删除(只删除从左到右第一个出现该值的元素)"""
        cur = self.__head
        pre = None
        while cur is not None:
            if cur.item == item:  # 找到了要删除的元素
                if cur == self.__head:  # 在头部找到了元素
                    self.__head = cur.next
                else:
                    pre.next = cur.next
                    # cur = pre.next 如果要删除全部item元素，则取消该注释，并注销下一行return
                return
            # 不是要找的元素，移动游标
            else:
                pre = cur
                cur = cur.next

    def reveres(self):
        """反转元素结点 """
        if self.is_empty() or self.length() <= 1:
            return
        j = 0
        while j < self.length() - 1:
            cur = self.__head
            for i in range(self.length() - 1):
                cur = cur.next
                if cur.next is None:
                    x = cur.item
                    self.remove(cur.item)
                    self.insert(j, x)
            j += 1


if __name__ == '__main__':
    singlinkList = SingleLinkList(10)
    print(singlinkList.is_empty())
    print(singlinkList.length())
    print("==================")
    singlinkList.append(15)
    print(singlinkList.is_empty())
    print(singlinkList.length())
    print("=================")
    singlinkList.add(20)
    singlinkList.add(10)
    singlinkList.add(40)
    singlinkList.append(50)
    singlinkList.insert(3, 80)
    singlinkList.travel()
    print("\n===================")
    singlinkList.remove(10)
    singlinkList.travel()

