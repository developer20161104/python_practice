from typing import List


class Solution:
    def climbStairs(self, n: int) -> int:
        # 状态转移方程为 dp[i] = dp[i-1] + dp[i-2]
        # 表示当前可选方案数等于前一步的方案加上前两步方案，之后走的一步或两步都是固定的
        # 流下了悔恨的泪水！
        f1, f2 = 1, 1
        for i in range(2, n+1):
            f1, f2 = f1 + f2, f1

        return f1

    def rob(self, nums: List[int]) -> int:
        # 状态转移方程为 dp[i] = max(dp[i-2]+cost[i], dp[i-1])
        # 表示选择方案只能出现在前一步的最大值或者前两步最大与当前的值的和之间
        if not len(nums):
            return 0
        # 在简化参数时稍微迷了点
        f1, f2 = nums[0], 0
        for i in range(1, len(nums)):
            f1, f2 = max(f2+nums[i], f1), f1

        return max(f1, f2)

    def isSubsequence(self, s: str, t: str) -> bool:
        for i in range(len(s)):
            try:
                cur_pos = t.index(s[i])
            # 找不到了必为False
            except ValueError as _:
                return False
            # 截取后半部分进行查找
            t = t[cur_pos+1:]

        return True

    def divisorGame(self, N: int) -> bool:
        # 纯数学思想
        return N % 2 == 0


class NumArray:
    # 状态转移方程为 dp[i] = nums[i-1] + dp[i-1]
    # 现将中间结果进行缓存，使用时再进行提取操作
    def __init__(self, nums: List[int]):
        lens = len(nums)
        self.dp = [0]*(lens + 1)
        self.dp[0] = 0

        for i in range(1, lens+1):
            self.dp[i] = nums[i-1] + self.dp[i-1]

    def sumRange(self, i: int, j: int) -> int:
        # 可以推导出来
        return self.dp[j+1] - self.dp[i]


if __name__ == '__main__':
    show = Solution()

    # 70 爬楼梯
    # print(show.climbStairs(4))

    # 198 打家劫舍
    # print(show.rob([2,7,9,3,1]))

    # 303 区域和检索-数组不可变
    # print(NumArray([-2, 0, 3, -5, 2, -1]).sumRange(2, 5))

    # 392 判断子序列
    # print(show.isSubsequence("acb","ahbgdc"))

    # 1025 除数博弈
    # print(show.divisorGame(100))
