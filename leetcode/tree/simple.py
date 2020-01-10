from typing import List


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def generate_tree(ele: List[int], pos: int) -> TreeNode:
    if pos >= len(ele) or ele[pos] is None:
        return None

    T = TreeNode(ele[pos])
    pos += 1
    T.left = generate_tree(ele, pos)
    pos += 1
    T.right = generate_tree(ele, pos)
    return T

def travel(T: TreeNode):
    if T is None:
        return
    print(T.val)
    travel(T.left)
    travel(T.right)


class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        #travel(p)
        travel(q)

        return True


if __name__ == '__main__':
    show = Solution()

    # 100 相同的树
    print(show.isSameTree(generate_tree([1,None,3], 0), generate_tree([1,2,3], 0)))
