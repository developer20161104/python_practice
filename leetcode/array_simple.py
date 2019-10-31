from typing import List
import math


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

    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # 使用k来统计已经出现的0的个数，用于后面的位移距离判断
        k = 0
        lens = len(nums)

        for i in range(lens):
            if nums[i] == 0:
                k += 1
            elif k > 0:
                nums[i - k] = nums[i]
                nums[i] = 0
        # print(nums)

    def thirdMax(self, nums: List[int]) -> int:
        # 使用一个大小为3的列表来收入前三个最大的数字
        # 表示无穷的技巧为使用float("-inf")
        lists = [float("-inf"), float("-inf"), float("-inf")]

        for i in nums:
            # 去掉重复
            if i in lists:
                continue

            if i > lists[0]:
                lists = [i, lists[0], lists[1]]
            elif i > lists[1]:
                lists = [lists[0], i, lists[1]]
            elif i > lists[2]:
                lists[2] = [lists[0], lists[1], i]

        # 判断语句还能与返回语句一起写的？
        return int(lists[0]) if math.isinf(lists[2]) else int(lists[2])

    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        # 利用了set的特性，但是空间复杂度不满足条件
        """
        lens = len(nums)
        return list(set(range(1, lens+1)) - set(nums))
        """
        # 顺着数值找索引
        # 将出现的数字作为下标，并对该位置加长度，然后再遍历这个数组，寻找低于这个长度的位置，即为消失的数字
        lists = []
        lens = len(nums)
        for i in range(lens):
            # 注意在添加长度后需要求余数，否则下标会溢出
            index = (nums[i] - 1) % lens
            nums[index] += lens

        for i in range(lens):
            if nums[i] <= lens:
                lists.append(i + 1)
        return lists

    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        # 设置计数变量来统计当前连续数量即可
        maxs, cur = 0, 0
        lens = len(nums)

        for i in nums:
            if i:
                cur += 1
            else:
                maxs = max(maxs, cur)
                cur = 0

        return max(cur, maxs)

    def findPairs(self, nums: List[int], k: int) -> int:
        # 过于耗时
        """
        # 排序后不需要再考虑前后互换重复，只需要判断相同元素重复即可
        lens, res = len(nums), 0
        # dicts保存的是当前已经加入的键值对
        nums, dicts = sorted(nums), {}
        # i为除最后一个外的每一个元素，j则为与i匹配的另一个元素
        for i in range(lens-1):
            j = i+1
            while j < lens and nums[j]-nums[i] < k:
                j += 1
            # j可能会越界
            if j < lens and nums[j]-nums[i] == k and nums[i] not in dicts:
                dicts[nums[i]] = nums[j]
                res += 1

        return res
        """
        # 用两个set,一个保存当前遍历的元素，一个保存满足条件的较小数字（自动过滤重复，妙呀）
        # 不能直接将所有元素添加，会把重复元素过滤造成错误，并且两个判断也不能合并，可能会出现同时满足的情况
        if k < 0:
            return 0
        diff, total = set(), set()
        for i in nums:
            # 当前值较大
            if i - k in total:
                diff.add(i - k)
            # 当前值较小
            if i + k in total:
                diff.add(i)
            total.add(i)

        return len(diff)

    def arrayPairSum(self, nums: List[int]) -> int:
        # 关键在于排序后将小的元素都放在一起，然后选取每对的第一个元素即可
        lens = len(nums)
        nums = sorted(nums)
        return sum(nums[i] for i in range(0, lens, 2))


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

    # 283 移动零
    # print(show.moveZeroes([0,0]))

    # 414 第三大的数
    # print(show.thirdMax([2,1,3]))

    # 448 找到数组中消失的数字
    # print(show.findDisappearedNumbers([4,3,2,7,8,2,3,1]))

    # 485 最大连续1个数
    # print(show.findMaxConsecutiveOnes([1,1,0,1,1,1]))

    # 532 数组中的K-diff对
    # print(show.findPairs([6,3,5,7,2,3,3,8,2,4], 2))

    # 561 数组拆分I
    # print(show.arrayPairSum([1,4,3,2]))
