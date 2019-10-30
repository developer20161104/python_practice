from typing import  List


class Solution:
    """
    # 同样使用了状态的枚举来进行处理，每个状态作为一个维度，然后写出状态转移方程与初始值即可
    # 同时可以在正确的前提下优化空间复杂度来实现O(1)的效果
    def maxProfit(self, prices: List[int]) -> int:
        lens = len(prices)
        if lens == 0:
            return 0

        # step1：initial varieties
        cur_own, cur_not = -prices[0], 0
        for i in range(lens):
            pre_not = cur_not
            # step2：get state transition equations
            cur_not = max(cur_not, cur_own + prices[i])
            cur_own = max(cur_own, pre_not - prices[i])

        # be sure of the final result
        return cur_not
    """
    # way 2
    # you can sell your stock anytime and prepare for the next one,
    # so add all positive subtract value
    def maxProfit(self, prices: List[int]) -> int:
        sub_sort = []
        lens = len(prices)

        # 获取差值列表
        for i in range(1, lens):
            sub_sort.append(prices[i] - prices[i-1])

        max_fit = 0
        for i in range(lens - 1):
            # 只要当前能得钱就放进来，类似贪心？
            if sub_sort[i] > 0:
                max_fit += sub_sort[i]

        return max_fit


if __name__ == '__main__':
    show = Solution()
    print(show.maxProfit([7,6,4,3,1]))