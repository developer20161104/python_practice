from typing import List


class Solution:

    # brute force exceeding time limit
    """
    def maxProfit(self, prices: List[int]) -> int:
        lens = len(prices)
        max_profit = 0

        for i in range(lens - 1):
            cur_pri = prices[i]

            for j in range(i + 1, lens):
                price_diff = prices[j] - cur_pri
                if price_diff > max_profit:
                    max_profit = price_diff

        return max_profit
    """
    # way 1
    # 作差后可以转化为最大连续子序列求取
    """
    def maxProfit(self, prices: List[int]) -> int:
        lens = len(prices)
        sub_sort = []
        for i in range(1, lens):
            sub_sort.append(prices[i] - prices[i-1])

        max_val = 0
        cur_val = 0
        for i in range(lens-1):
            cur_val += sub_sort[i]
            # 状态转移方程： max(cur_val + max_val, max_val)
            if cur_val < 0:
                cur_val = 0

            max_val = max(max_val, cur_val)

        return max_val
    """
    # way 2
    # dp enumerate
    # 状态转移方程分别为
    # 当前不持股票 = max(之前也不持股票， 之前有股票但是卖出去了)
    # 当前持股票 = max(之前也持股票， 之前不持股票但是现在买了这股)
    def maxProfit(self, prices: List[int]) -> int:
        lens = len(prices)
        if lens == 0:
            return 0
        # 初始时，不持股票为0，持有股票则为购进第一支价格的相反数
        max_own, max_not = -prices[0], 0

        for i in range(lens):
            max_not = max(max_not, max_own + prices[i])
            max_own = max(max_own, -prices[i])

        # 最后肯定不持股票得到的收益更多
        return max_not

if __name__ == '__main__':
    show = Solution()
    print(show.maxProfit([7,1,5,3,6,4]))