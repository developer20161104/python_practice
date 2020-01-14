from typing import List


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


# 采用先序遍历 将列表转化为二叉树
def generate_tree(ele: List[int], pos: int) -> TreeNode:
    # 有问题：在前面的父节点包含一个为空时，计算会有误(即列表元素少于2**h-1时)
    # 样本问题
    # example: [0,2,4,1,None,3,-1,5,1,None,6,None,8]

    if pos >= len(ele) or ele[pos] is None:
        return None

    T = TreeNode(ele[pos])
    # 创建左右节点的下标需要注意
    # 之前一直有问题就是出在下标的计算上
    T.left = generate_tree(ele, 2 * pos + 1)
    T.right = generate_tree(ele, 2 * pos + 2)
    return T


# 先序输出
def travel(T: TreeNode):
    if T is None:
        return
    print(T.val)
    travel(T.left)
    travel(T.right)


def BFS(T: TreeNode):
    from collections import deque
    # 队列实现
    q = deque()
    q.append(T)
    lens = 0
    while len(q):
        lens += 1
        size = len(q)
        # BFS模板系列，自己写的根本没照顾到层次问题
        for i in range(size):
            # 逐层进行遍历与添加元素
            ele = q.popleft()
            if ele.left:
                q.append(ele.left)
            if ele.right:
                q.append(ele.right)

    return lens


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
        # 中序遍历结果思路有问题
        """
        def get_val(roots: TreeNode, arr: List[int]):
            if roots is None:
                arr.append(None)
                return None
            get_val(roots.left, arr)
            arr.append(roots.val)
            get_val(roots.right, arr)

        arrs = []
        get_val(root, arrs)
        return arrs[:arrs.index(root.val)] == arrs[len(arrs)-1:arrs.index(root.val):-1]
        """

        def judge(T1: TreeNode, T2: TreeNode) -> bool:
            # 首先判断结构是否一致
            if T2 == T1:
                return True
            # 再此基础上判断值是否相等
            if (T1 is None or T2 is None) or T1.val != T2.val:
                return False

            # 将结果汇总即可
            return judge(T1.left, T2.right) and judge(T1.right, T2.left)

        return judge(root.left, root.right) if root else True

    def maxDepth(self, root: TreeNode) -> int:
        # 参数式递归
        """
        def get_deep(roots: TreeNode, lens) -> int:
            if roots is None:
                # 将结果返回
                return lens
            # 每层都要加 1
            lens += 1
            return max(get_deep(roots.left, lens), get_deep(roots.right, lens))
        return get_deep(root, 0)
        """
        # 方法二：直观递归
        # 只看结束条件即可
        if root is None:
            return 0
        # 增加的因子直接放到判断返回中，进行简化
        return max(Solution().maxDepth(root.left), Solution().maxDepth(root.right))+1

    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        """
        def create_SearchTree(num: List[int], st: int, end: int) -> TreeNode:
            if st > end:
                return None

            # 思路不错，总是有些想不到
            mid = st + (end-st)//2
            # 每次利用中间位置构造中间节点，两边则为左右节点
            T = TreeNode(num[mid])
            T.left = create_SearchTree(num, st, mid-1)
            T.right = create_SearchTree(num, mid+1, end)

            return T

        return create_SearchTree(nums, 0, len(nums)-1)
        """
        # 更适合递归的解法
        if nums:
            # 右移一位，即除2
            m = len(nums) >> 1
            T = TreeNode(nums[m])
            T.left, T.right = map(self.sortedArrayToBST, [nums[:m], nums[m+1:]])
            return T


if __name__ == '__main__':
    show = Solution()

    # 100 相同的树
    # print(show.isSameTree(generate_tree([1,2,3,4], 0), generate_tree([1,2,3,None,4], 0)))

    # 101 对称二叉树
    # print(show.isSymmetric(generate_tree([1,2,3], 0)))

    # 104 二叉树的最大深度
    # print(show.maxDepth(generate_tree([3,9,20,None,None,15,7], 0)))

    # 108 将有序数组转化为二叉搜索树
    # print(travel(show.sortedArrayToBST([-10,-3,0,5,9])))
