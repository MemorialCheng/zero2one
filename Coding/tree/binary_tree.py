# -*- coding:utf-8 -*-
"""
@Project ：zero2one 
@File ：binary_tree.py
@Author ：cheng
@Date ：2021/1/26
@Description : 
"""


class Node(object):
    def __init__(self, item):
        self.elem = item
        self.lchild = None
        self.rchild = None


class BinaryTree(object):
    def __init__(self):
        self.root = None

    def add(self, item):
        node = Node(item)
        if self.root is None:
            self.root = node
            return
        queue = [self.root]
        while queue:
            cur_node = queue.pop(0)
            if cur_node.lchild is None:
                cur_node.lchild = node
                return
            else:
                queue.append(cur_node.lchild)
            if cur_node.rchild is None:
                cur_node.rchild = node
                return
            else:
                queue.append(cur_node.rchild)

    def breadth_Search(self):
        """
        广度遍历
        :return:
        """
        if self.root is None:
            return
        queue = [self.root]
        while queue:
            cur_node = queue.pop(0)
            print(cur_node.elem, end=" ")
            if cur_node.lchild is not None:
                queue.append(cur_node.lchild)
            if cur_node.rchild is not None:
                queue.append(cur_node.rchild)

    def preorder(self, node):
        """
        先序遍历,递归实现
        :return:
        """
        if node is None:
            return
        print(node.elem, end=" ")
        self.preorder(node.lchild)
        self.preorder(node.rchild)

    def inorder(self, node):
        """
        中序遍历,递归实现
        :return:
        """
        if node is None:
            return
        self.inorder(node.lchild)
        print(node.elem, end=" ")
        self.inorder(node.rchild)

    def postorder(self, node):
        """
        后序遍历,递归实现
        :return:
        """
        if node is None:
            return
        self.postorder(node.lchild)
        self.postorder(node.rchild)
        print(node.elem, end=" ")


if __name__ == '__main__':
    binary_tree = BinaryTree()
    binary_tree.add(0)
    binary_tree.add(1)
    binary_tree.add(2)
    binary_tree.add(3)
    binary_tree.add(4)
    binary_tree.add(5)
    binary_tree.add(6)
    binary_tree.add(7)
    binary_tree.add(8)
    binary_tree.add(9)
    print("\n==========广度遍历============")
    binary_tree.breadth_Search()
    print("\n==========先序遍历============")
    binary_tree.preorder(binary_tree.root)
    print("\n==========中序遍历============")
    binary_tree.inorder(binary_tree.root)
    print("\n==========后序遍历============")
    binary_tree.postorder(binary_tree.root)
