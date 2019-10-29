from typing import List


class Solution:
    # 双指针法
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        lens = len(numbers)
        i, j = 0, lens - 1

        # 由于题给条件，因此只能有如下情况
        while i != j:
            ans = numbers[i] + numbers[j]
            if ans > target:
                j -= 1
            elif ans < target:
                i += 1
            else:
                return [i + 1, j + 1]

        return []

    def majorityElement(self, nums: List[int]) -> int:

        # way 1 先排序，再判断
        """
        nums = sorted(nums)
        lens = len(nums)
        max_num, max_time = nums[0], 0
        cur_num, cur_time = max_num, max_time
        for i in range(lens):
            if cur_num == nums[i]:
                cur_time += 1
            else:
                if cur_time > max_time:
                    max_time = cur_time
                    max_num = cur_num
                cur_num = nums[i]
                cur_time = 1
        # 最后还需要进行一轮判断
        if cur_time > max_time:
            return cur_num
        else:
            return max_num
        """

        # way 2 摩尔投票法，正负判断
        lens = len(nums)
        candidate = nums[0]
        moer = 1

        for i in range(1, lens):
            # 必须先判断moer变量的值，再判断当前待定值
            if moer == 0:
                candidate = nums[i]
                moer = 1
            elif candidate == nums[i]:
                moer += 1
            else:
                moer -= 1

        return candidate

    def containsDuplicate(self, nums: List[int]) -> bool:
        # 利用set中无重复元素特性
        return len(set(nums)) != len(nums)

    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        # 维护一个字典即可
        dict = {}
        lens = len(nums)
        for i in range(lens):
            if nums[i] in dict and i - dict[nums[i]] <= k:
                return True
            # 存在性问题，不管在不在都得更新
            dict[nums[i]] = i

        return False

    def missingNumber(self, nums: List[int]) -> int:
        # way 1: wrong answer
        # exceeding time limit
        """
        lens = len(nums)
        return [i for i in range(lens+1) if i not in nums][0]
        """

        # 利用set性质以及并集结果
        """
        lens = len(nums)
        return (set(range(lens+1)) - set(nums)).pop()
        """

        # way 2 use math
        lens = len(nums)
        return lens * (lens + 1) // 2 - sum(nums)


if __name__ == '__main__':
    show = Solution()

    # 167 两数和（有序数组）
    # print(show.twoSum([2, 7, 11, 15], 9))

    # 168 求众数
    # print(show.majorityElement([47,47,72,47,72,47,79,47,12,92,13,47,47,83,33,15,18,47,47,47,47,64,47,65,47,47,47,47,70,47,47,55,47,15,60,47,47,47,47,47,46,30,58,59,47,47,47,47,47,90,64,37,20,47,100,84,47,47,47,47,47,89,47,36,47,60,47,18,47,34,47,47,47,47,47,22,47,54,30,11,47,47,86,47,55,40,49,34,19,67,16,47,36,47,41,19,80,47,47,27]))

    # 217 存在重复元素
    # print(show.containsDuplicate([1,2,3,1]))

    # 219 存在重复元素2
    # print(show.containsNearbyDuplicate([1,2,3,1], 3))

    # 268 缺失数字
    # print(show.missingNumber([9,6,4,2,3,5,7,0,1]))
