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


if __name__ == '__main__':
    show = Solution()

    # 70 爬楼梯
    print(show.climbStairs(4))

    # 198 打家劫舍
    print(show.rob([2,7,9,3,1]))