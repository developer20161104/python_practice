from typing import List


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


# 采用先序遍历 将列表转化为二叉树
def generate_tree(ele: List[int], pos=0) -> TreeNode:
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
    def __init__(self):
        # 538 统计当前累加值
        self.total = 0

        # 543 二叉树直径
        # 取1时可将空节点也一并考虑
        self.res = 1

        # 563 二叉树坡度
        self.total_degree = 0

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
        return max(Solution().maxDepth(root.left), Solution().maxDepth(root.right)) + 1

    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []

        from collections import deque
        # 保存每级的 val
        stack = []
        q = deque()
        q.append(root)

        while q:
            floor_size = len(q)
            temp = []
            for i in range(floor_size):
                pops = q.popleft()
                if pops is not None:
                    # 由于结构体约束，还是得在弹出时添加，不增加额外时间消耗
                    temp.append(pops.val)
                    if pops.left:
                        q.append(pops.left)
                    if pops.right:
                        q.append(pops.right)
            stack.append(temp)
        return stack[::-1]

    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        # 此种解法不能根据大小判断，因此得到的结果不是二叉搜索树
        """lens = len(nums)
        if not lens:
            return None
        pos = lens//2
        T = TreeNode(nums[pos])
        T.left = generate_tree(nums[:pos], 0)
        T.right = generate_tree(nums[pos+1:], 0)

        return T"""

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
            T.left, T.right = map(self.sortedArrayToBST, [nums[:m], nums[m + 1:]])
            return T

    def isBalanced(self, root: TreeNode) -> bool:
        def get_len(roots: TreeNode) -> int:
            if not roots:
                return 0
            # 当前已经出现不满足平衡二叉树的就直接全部返回-1，剪枝操作
            left = get_len(roots.left)
            if left == -1:
                return -1
            right = get_len(roots.right)
            if right == -1:
                return -1

            # 需要比较的都是最大高度，因此需要 max
            return max(left, right) + 1 if abs(left - right) < 2 else -1

        return get_len(root) != -1

    def minDepth(self, root: TreeNode) -> int:
        # BFS 大法好
        """
        if not root:
            return 0
        from collections import deque
        q = deque()
        q.append(root)
        height = 0
        while q:
            # 层数统计
            height += 1
            size = len(q)
            for i in range(size):
                ele = q.popleft()
                if ele:
                    if ele.left:
                        q.append(ele.left)
                    if ele.right:
                        q.append(ele.right)
                    # 当其左右子树皆不存在时，此时即为最小高度
                    if ele.left is None and ele.right is None:
                        return height

        return height"""
        # 尝试递归
        # 关键在于对结束条件的把握
        # 当前节点为空时不返回高度
        if not root:
            return 0
        # 当前节点无子节点时返回高度（可与包含一个节点的合并）
        """
        if root.left is None and root.right is None:
            return 1
        """
        # 当前节点包含一个节点时继续向下寻找（此处没想到）
        if root.left is None or root.right is None:
            return self.minDepth(root.left) + self.minDepth(root.right) + 1
        # 两个都有子节点时找短的部分
        else:
            return min(self.minDepth(root.left), self.minDepth(root.right)) + 1

    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        # 排除无节点但是sum为0的情况
        if not root:
            return False

        sum -= root.val
        # 如果已经为叶子节点，则进行判断即可
        if root.left is None and root.right is None:
            return True if not sum else False

        # 注意到只要有一个满足即可
        return self.hasPathSum(root.left, sum) or self.hasPathSum(root.right, sum)

    def invertTree(self, root: TreeNode) -> TreeNode:
        # 实质为后序递归法
        if not root:
            return None

        # 先分别获得左右子树并异位
        left = self.invertTree(root.right)
        right = self.invertTree(root.left)

        # 再将子树与根节点拼接
        root.left = left
        root.right = right
        return root

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # 此处为利用搜索树的性质解决问题
        if root.val > p.val and root.val > q.val:
            return self.lowestCommonAncestor(root.left, p, q)
        elif root.val < p.val and root.val < q.val:
            return self.lowestCommonAncestor(root.right, p, q)
        # 当p与q分别在两侧时，此时的节点为公共节点
        else:
            return root

    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        """
        total = []
        L = []

        def get_path(roots: TreeNode):
            if not roots:
                return []

            L.append(str(roots.val))

            if roots.left is None and roots.right is None:
                total.append("->".join(L))
            else:
                get_path(roots.left)
                get_path(roots.right)
            # 回溯思想
            L.remove(L[-1])

        get_path(root)
        return total
        """
        if not root:
            return []

        # 当前已经到叶子节点后
        if root.left is None and root.right is None:
            return [str(root.val)]

        path = []
        # 递归接收各个路径
        if root.left:
            for i in self.binaryTreePaths(root.left):
                path.append(str(root.val) + "->" + i)
        if root.right:
            for i in self.binaryTreePaths(root.right):
                path.append(str(root.val) + "->" + i)
        return path

    def sumOfLeftLeaves(self, root: TreeNode) -> int:
        if root is None:
            return 0
        """
        # 采用中序遍历
        sums = self.sumOfLeftLeaves(root.left)
        # 左右相对于父节点来考虑，因此需要预先判别
        if root.left and root.left.left is None and root.left.right is None:
            # 此处的sums增加的是右侧左叶子节点的值
            sums += root.left.val

        sums += self.sumOfLeftLeaves(root.right)
        return sums
        """
        # 改写变量
        if root.left and not root.left.left and not root.left.right:
            # 注意到此处为 right 部分，毕竟条件已经将左边部分约束了
            return root.left.val + self.sumOfLeftLeaves(root.left)
        else:
            # 然后把两边的左叶子节点统计起来
            return self.sumOfLeftLeaves(root.left) + self.sumOfLeftLeaves(root.right)

    counts = 0

    def pathSum(self, root: TreeNode, sum: int) -> int:
        if root is None:
            return 0

        # 内部判别函数
        def judge_path(roots: TreeNode, sums: int):
            if roots is None:
                return 0

            # 注意到此处只能使用变量统计，如果直接返回会少了同根下多个的情况
            sums -= roots.val
            count = 0
            if not sums:
                count += 1
            # 统计左右两边
            return judge_path(roots.left, sums) + judge_path(roots.right, sums) + count

        # 递归统计
        counts = judge_path(root, sum)
        return counts + self.pathSum(root.left, sum) + self.pathSum(root.right, sum)

    def findMode(self, root: TreeNode) -> List[int]:
        # 思路有误，左侧子树不一定与当前根节点的值相等
        """
        if not root:
            return []

        cur_max = 0
        max_list = []
        while root:
            cur_root, count = root, 1
            while cur_root.left:
                count ++= 1
                cur_root = cur_root.left

            if count >= cur_max:
                cur_max = count
                max_list.append([count, root.val])

            root = root.right

        com = max(max_list, key=lambda x: x[0])[0]
        return [x[1] for x in max_list if x[0] == com]
        """

        arr = []

        # 遍历获取结果
        def get_list(roots: TreeNode):
            if not roots:
                return
            get_list(roots.left)
            arr.append(roots.val)
            get_list(roots.right)

        get_list(root)
        if not arr:
            return []

        # 使用了内库Counter来构建字典
        from collections import Counter
        dic = Counter(arr)
        com = max(dic.values())

        # 列表推导式来生成结果
        return [x[0] for x in dic.items() if x[1] == com]

    def getMinimumDifference(self, root: TreeNode) -> int:
        arr = []

        # 遍历获取结果
        def get_list(roots: TreeNode):
            if not roots:
                return
            get_list(roots.left)
            arr.append(roots.val)
            get_list(roots.right)

        get_list(root)
        if not root:
            return None

        """
        cur_min = arr[1] - arr[0]
        
        for i in range(1, len(arr)):
            if cur_min > arr[i] - arr[i-1]:
                cur_min = arr[i] - arr[i-1]

        return cur_min
        """
        # 缩写
        return min(map(lambda x, y: y - x, arr[:len(arr)], arr[1:]))

    def convertBST(self, root: TreeNode) -> TreeNode:
        # 遍历两遍慢出天际
        """
        if not root:
            return None

        arr = []

        def get_lists(roots: TreeNode, choose: int):
            if not roots:
                return

            get_lists(roots.left, choose)
            if choose:
                arr.append(roots.val)
            else:
                roots.val += sum(arr[arr.index(roots.val)+1:])
            get_lists(roots.right, choose)

        get_lists(root, 1)
        get_lists(root, 0)
        return root
        """
        # 遍历顺序：右中左
        if not root:
            return root
        if root.right:
            self.convertBST(root.right)
            # 其他位置判别
            self.total += root.val
            root.val = self.total
            self.convertBST(root.left)
        else:
            # 尾部（右子树位置）判别
            root.val += self.total
            # 注意到total起始为0，因此
            self.total = root.val
            self.convertBST(root.left)
        return root

    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        # 题意可以转化为求最长左右子树长度
        def get_len(roots: TreeNode) -> int:
            if not roots:
                return 0
            L = get_len(roots.left)
            R = get_len(roots.right)
            # 在此处判断每个节点左右子树长度，求其最大值
            self.res = max(self.res, L + R + 1)
            return max(L, R) + 1

        get_len(root)
        # 由于最终返回的结果不一致，因此需要内嵌函数
        return self.res - 1

    def findTilt(self, root: TreeNode) -> int:
        if not root:
            return 0

        def get_degree(roots: TreeNode):
            if not roots:
                return 0

            L = get_degree(roots.left)
            R = get_degree(roots.right)
            # 求取各个坡度
            self.total_degree += abs(L-R)
            # 错误之处在此，我用的是差值，真实结果应该是 相加
            return L + R + roots.val

        # 递归还是嘿麻烦
        get_degree(root)
        return self.total_degree


if __name__ == '__main__':
    show = Solution()

    # 100 相同的树
    # print(show.isSameTree(generate_tree([1,2,3,4], 0), generate_tree([1,2,3,None,4], 0)))

    # 101 对称二叉树
    # print(show.isSymmetric(generate_tree([1,2,3], 0)))

    # 104 二叉树的最大深度
    # print(show.maxDepth(generate_tree([3,9,20,None,None,15,7], 0)))

    # 107 二叉树的层次遍历II
    # print(show.levelOrderBottom(generate_tree([3,9,20,None,None,15,7], 0)))

    # 108 将有序数组转化为二叉搜索树
    # print(show.sortedArrayToBST([-10,-3,0,5,9]))

    # 110 平衡二叉树
    # print(show.isBalanced(generate_tree([1,None,2,None,None,3,None], 0)))

    # 111 二叉树的最小深度
    # print(show.minDepth(generate_tree([3,9,20,None,None,15,7], 0)))

    # 112 路径总和
    # print(show.hasPathSum(generate_tree([], 0), 0))

    # 226 翻转二叉树
    # print(show.invertTree(generate_tree([4,2,7,1,3,6,9], 0)))

    # 235 二叉搜索树的最近公共祖先
    # print(show.lowestCommonAncestor(generate_tree([6,2,8,0,4,7,9,None,None,3,5], 0), TreeNode(2), TreeNode(8)).val)

    # 257 二叉树的所有路径
    # print(show.binaryTreePaths(generate_tree([1,2,3,None,5], 0)))

    # 404 左叶子之和
    # print(show.sumOfLeftLeaves(generate_tree([3,9,20,None,None,15,7], 0)))

    # 437 路径总和III
    # print(show.pathSum(generate_tree([1,-2,-3,1,3,-2,None,-1]), -1))

    # 501 二叉搜索树中的众数
    # print(show.findMode(generate_tree([])))

    # 530 二叉搜索树的最小绝对差
    # print(show.getMinimumDifference(generate_tree([1,None,3,None,None,2,None])))

    # 538 把二叉搜索树转换为累加树
    # print(travel(show.convertBST(generate_tree([5,2,13]))))

    # 543 二叉树的直径
    # print(show.diameterOfBinaryTree(generate_tree([1,2,3,4,5])))

    # 563 二叉树的坡度
    # print(show.findTilt(generate_tree([1,2,None,3,4])))
