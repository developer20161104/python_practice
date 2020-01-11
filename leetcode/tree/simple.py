from typing import List


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


# 采用先序遍历 将列表转化为二叉树
def generate_tree(ele: List[int], pos: int) -> TreeNode:
    if pos >= len(ele) or ele[pos] is None:
        return None

    T = TreeNode(ele[pos])
    # 创建左右节点的下标需要注意
    # 之前一直有问题就是出在下标的计算上
    T.left = generate_tree(ele, 2*pos+1)
    T.right = generate_tree(ele, 2*pos+2)
    return T


def travel(T: TreeNode):
    if T is None:
        return
    print(T.val)
    travel(T.left)
    travel(T.right)


class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        # 需要判断是否为None
        # 只有当两者都为None时，才为True
        if p is None:
            return True if q is None else False

        # 少了判断q为None的条件
        if q is None or p.val != q.val:
            return False

        # 递归判断注意实例化类
        res1 = Solution().isSameTree(p.left, q.left)
        res2 = Solution().isSameTree(p.right, q.right)
        return res1 & res2

    def isSymmetric(self, root: TreeNode) -> bool:
        pass


if __name__ == '__main__':
    show = Solution()

    # 100 相同的树
    # print(show.isSameTree(generate_tree([1,2,3,4], 0), generate_tree([1,2,3,None,4], 0)))

    # 101 对称二叉树
    print(show.isSymmetric(generate_tree([1,2,2,3,4,4,3], 0)))