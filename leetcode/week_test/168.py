from typing import List


class Solution:
    def findNumbers(self, nums: List[int]) -> int:
        count = 0
        for d in nums:
            if not len(str(d)) % 2:
                count += 1

        return count

    def isPossibleDivide(self, nums: List[int], k: int) -> bool:
        # 连续数字的含义都没理解
        """
        if len(nums) % k:
            return False
        nums = sorted(nums) + [0]
        pre, count = nums[0], 1
        for ch in nums[1:]:
            if ch == pre:
                count += 1
            else:
                if count > k:
                    return False
                pre = ch
                count = 1

        return True
        """


if __name__ == '__main__':
    show = Solution()

    # 偶数位个数
    # print(show.findNumbers([1,23,213,1243]))

    # 划分数组为连续数字集合
    # print(show.isPossibleDivide([1,1,1,1,2,2], 3))
