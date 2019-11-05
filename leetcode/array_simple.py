from typing import List
from functools import reduce
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
        dicts = {}
        lens = len(nums)
        for i in range(lens):
            if nums[i] in dicts and i - dicts[nums[i]] <= k:
                return True
            # 存在性问题，不管在不在都得更新
            dicts[nums[i]] = i

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

    def matrixReshape(self, nums: List[List[int]], r: int, c: int) -> List[List[int]]:
        # 先排个列表再进行处理，有点耗时
        lists = [x for l in nums for x in l]
        lens = len(lists)
        if r * c - lens:
            return nums

        return [lists[i * c:(i + 1) * c] for i in range(r)]

    def findUnsortedSubarray(self, nums: List[int]) -> int:

        # wrong answer: 无法判断重复数组，思路不对
        """
        i, j = 1,  len(nums)-1
        if j:
            while i < j and nums[i] >= nums[i - 1]:
                i += 1
            while j > 0 and nums[j] >= nums[i-1] and nums[j] >= nums[j - 1]:
                j -= 1
        else:
            return 0
        return 0 if i == len(nums)-1 and j else j-i+2
        """
        # 对于重复序列无效
        """
        lens = len(nums)
        if lens < 2:
            return 0
        max_front = nums[0]
        for i in range(lens-1):
            if nums[i] > nums[i+1]:
                max_front = i
                break
        max_back = nums[lens-1]
        for i in range(lens-1, 0, -1):
            if nums[i] < nums[i-1] or (max_front < lens and nums[i] < nums[max_front]):
                max_back = i
                break
        if max_front == nums[0] and max_back == nums[lens-1]:
            return 0
        return max_back-max_front+1
        """
        # 使用排序对比，过滤各种花里胡哨的样例
        lens = len(nums)
        left, right = lens - 1, 0
        nums_sort = sorted(nums)
        for i in range(lens):
            if nums[i] != nums_sort[i]:
                left = i
                break

        for i in range(lens - 1, -1, -1):
            if nums[i] != nums_sort[i]:
                right = i
                break

        return 0 if left == lens - 1 and right == 0 else right - left + 1

    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        # 对于两边都做了特殊判断，其实可以考虑防御式编程思想，判断之前就在首尾各加一个0，排除特殊情况
        def judge(cur_zeros: int, n: int):
            jud = cur_zeros % 2
            if jud:
                n -= cur_zeros // 2
            elif cur_zeros:
                n -= cur_zeros // 2 - 1
            return n

        lens = len(flowerbed)
        cur_zero = 1 if not flowerbed[0] else 0
        for i in flowerbed:
            if not i:
                cur_zero += 1
            else:
                n = judge(cur_zero, n)
                cur_zero = 0

        if not flowerbed[lens - 1]:
            cur_zero += 1
            n = judge(cur_zero, n)
        return False if n > 0 else True

    def maximumProduct(self, nums: List[int]) -> int:
        lens = len(nums)
        nums = sorted(nums)

        # 没必要进行判断
        """
        lefts = 0
        for i in nums:
            if i < 0:
                lefts += 1

        right = nums[lens-2]*nums[lens-3]
        if lefts < 2:
            return right*nums[lens-1]
        else:
            left = nums[0] * nums[1]
            return left*nums[lens-1] if left > right else right*nums[lens-1]
        """
        return max(reduce(lambda a, b: a * b, nums[lens - 3:]), nums[0] * nums[1] * nums[lens - 1])

    def findMaxAverage(self, nums: List[int], k: int) -> float:
        # 暴力枚举超时，每次都要全部求和会超时
        """
        total = -10000000
        times = len(nums)-k+1
        for i in range(times):
            total = max(total, sum(nums[i:i+k]))

        return total
        """
        # total仅记录当前的max，cur则为移动窗口，注意两者区别！
        total = sum(nums[:k])
        cur = total
        times = len(nums) - k + 1
        for i in range(1, times):
            if k == 1:
                total = max(total, nums[i])
            else:
                # 相比直接全体求和，保持一个窗口会大大减少求和时间
                cur = cur - nums[i - 1] + nums[i + k - 1]
                total = max(total, cur)

        return total / k

    def imageSmoother(self, M: List[List[int]]) -> List[List[int]]:
        row, col = len(M), len(M[0])
        # 对于图类型的题目，常采用偏移数组来进行位移，减少不必要的条件判断
        move_r = [-1, 1, 0, 0, -1, -1, 1, 1]
        move_c = [0, 0, -1, 1, -1, 1, -1, 1]
        ans = []

        for i in range(row):
            cur_row = []
            for j in range(col):
                count = 1
                sums = M[i][j]
                for k in range(8):
                    # 预判下一步是否符合条件
                    next_r, next_c = i + move_r[k], j + move_c[k]
                    # python竟然阔以这么玩
                    if (0 <= next_r < row) and (0 <= next_c < col):
                        count += 1
                        sums += M[next_r][next_c]
                cur_row.append(sums // count)
            ans.append(cur_row)

        return ans

    def checkPossibility(self, nums: List[int]) -> bool:
        # 注意区分两种情况：2,3,3,1,4 与 1,2,8,2,4 有点麻烦
        count, lens = 0, len(nums)
        if lens < 2:
            return True
        if nums[0] > nums[1]:
            nums[0] = nums[1]
            count += 1

        for i in range(1, lens - 1):
            if nums[i] > nums[i + 1]:
                count += 1
                if count > 1:
                    return False
                # 败在此处逻辑，只有两种情况：把大的变小，把小的变大
                # 更换逻辑则为从右向左选择替换
                if nums[i + 1] < nums[i - 1]:
                    nums[i + 1] = nums[i]
                else:
                    nums[i] = nums[i - 1]

        return True

    def findLengthOfLCIS(self, nums: List[int]) -> int:
        maxs, cur, lens = 0, 1, len(nums)
        if not lens:
            return 0
        # 保存当前最大的记录即可
        for i in range(lens - 1):
            if nums[i] < nums[i + 1]:
                cur += 1
            else:
                maxs = max(maxs, cur)
                cur = 1

        return max(maxs, cur)

    def findShortestSubArray(self, nums: List[int]) -> int:
        # 超时,使用dict没错，但是统计次数的话后面还要重复查找，费时
        """
        dicts = {}
        for i in nums:
            if i not in dicts:
                dicts[i] = 1
            else:
                dicts[i] += 1

        # 通过value获取对应的key值
        # val = list(dicts.keys())[list(dicts.values()).index(max(dicts.values()))]
        max_val = max(dicts.values())
        pre_dig = [i for i in dicts if dicts[i] == max_val]
        min_len, lens = 50001, len(nums)
        for key in pre_dig:
            i, j = 0, lens-1
            while nums[i] != key:
                i += 1
            while nums[j] != key:
                j -= 1
            min_len = min(min_len, j-i+1)

        return min_len
        """
        # 使用dict保存每个元素出现位置，则无需后面的位置判别
        # 以空间换时间
        lens = len(nums)
        dicts = {}
        for i in range(lens):
            if nums[i] not in dicts:
                dicts[nums[i]] = [i]
            else:
                dicts[nums[i]].append(i)

        # 找出子列表中长度最长值
        max_num = max([len(x) for x in dicts.values()])
        min_len = lens
        for i in dicts:
            if len(dicts[i]) == max_num:
                # 有了每个元素的位置，只需将最后一个与第一个作差即可得到最终结果
                min_len = min(min_len, dicts[i][-1] - dicts[i][0] + 1)

        return min_len

    def isOneBitCharacter(self, bits: List[int]) -> bool:
        # 不能投机取巧
        """
        lens = len(bits)
        if lens < 2:
            return True
        if bits[-1] == bits[-2] == 0:
            return True
        elif lens > 2 and bits[-3]:
            return True
        return False
        """
        lens, i = len(bits), 0
        while i < lens:
            # 出现为1时抵消两个
            if bits[i]:
                i += 2
            else:
                # 如果最后只剩一个，则必为True
                if i == lens - 1:
                    return True
                i += 1

        return False

    def pivotIndex(self, nums: List[int]) -> int:
        # 由于有负数出现，因此无法以大小移动指针
        # 当出现正负数时候，不能直接以大小来进行衡量，因此可以考虑采用窗口包裹，简化实现
        """
        lens = len(nums)
        if lens < 3:
            return -1
        left_val, right_val = nums[0], nums[-1]
        left, right = 0, lens-1

        while left+1 < right:
            if left_val > right_val:
                right -= 1
                right_val += nums[right]
            elif left_val < right_val:
                left += 1
                left_val += nums[left]
            else:
                right -= 1
                right_val += nums[right]
                left += 1
                left_val += nums[left]

        return left if left_val == right_val and left+1 != right else -1
        """
        # 反正要移动指针，不如一开始就分成两块，直到找到相等的即可，窗口思想
        lens = len(nums)
        if lens < 3:
            return -1
        # 初始时左边为空，右边为从第一项开始的求和
        left, right = 0, sum(nums[1:])

        # 只能从最前面开始，否则会漏项 left=null，right=all=0
        for i in range(0, lens - 1):
            if left == right:
                return i
            left += nums[i]
            right -= nums[i + 1]
        # 必须把最后的盒子也要算上 left=all=0，right=null
        if left == right:
            return lens - 1
        return -1


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

    # 566 重塑矩阵
    # print(show.matrixReshape([[1,2],[3,4]],2,4))

    # 581 最短无序连续子数组（难）
    # print(show.findUnsortedSubarray([3,2]))

    # 605 种花问题（防御式编程思想）
    # print(show.canPlaceFlowers([0,0,1,0,0,0,1,0,0], 2))

    # 628 三个数的最大乘积
    # print(show.maximumProduct([1,2,3,4]))

    # 643 子数组最大平均数I
    # print(show.findMaxAverage([4,2,1,3,3],2))

    # 661 图片平滑器
    # print(show.imageSmoother([[1,1,1],[1,0,1],[1,1,1]]))

    # 665 非递减数列（难）
    # print(show.checkPossibility([1,2,8,1,6]))

    # 674 最大连续递增序列
    # print(show.findLengthOfLCIS([2,2,2]))

    # 697 数组的度（需要优化）
    # print(show.findShortestSubArray([1, 2, 2, 3, 1]))

    # 717 1比特与2比特字符
    # print(show.isOneBitCharacter([0,1,1,0]))

    # 724 寻找数组的中心索引
    # print(show.pivotIndex([-1,-1,-1,0,1,1]))
