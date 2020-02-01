from typing import List


class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        # 空间复杂度为O(n)，时间复杂度为O(n)
        """
        p = set()
        for dig in nums:
            if dig in p:
                p.remove(dig)
            else:
                p.add(dig)

        return p.pop()
        """

        # 两个数之间的关系——异或
        # 时间复杂度O(n)，空间复杂度为O(1)
        """
        from functools import reduce
        return reduce(lambda x, y: x ^ y, nums)
        """
        # 异或大法好
        # 不调用库函数直接逐个异或
        cur = nums[0]
        for i in nums[1:]:
            cur ^= i
        return cur


if __name__ == '__main__':
    show = Solution()

    # 136 只出现一次的数字
    # print(show.singleNumber([4, 1, 2, 1, 2]))
