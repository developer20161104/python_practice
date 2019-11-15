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

    def fib(self, N: int) -> int:
        pre_1, pre_2 = 0, 1
        if not N:
            return pre_1
        elif N == 1:
            return pre_2
        for i in range(2, N + 1):
            cur_val = pre_1 + pre_2
            pre_1 = pre_2
            pre_2 = cur_val

        return pre_2

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

    def minCostClimbingStairs(self, cost: List[int]) -> int:
        # 动态规划还是有些想不出来
        # 空间复杂度还可以优化
        """lens = len(cost)
        cur_cost = [0]*lens
        cur_cost[0], cur_cost[1] = cost[0], cost[1]
        for i in range(2, lens):
            # 记录之前的数值，方便后续使用，以空间换时间
            cur_cost[i] = min(cur_cost[i-1]+cost[i], cur_cost[i-2]+cost[i])

        return min(cur_cost[-1], cur_cost[-2])"""
        # 只需要保存左右两个值即可
        # 状态转移方程为 f(n) = min(f(n-1)+cost[n-1], f(n-2)+cost[n-2])
        lens = len(cost)
        left, right = 0, 0
        for i in range(2, lens + 1):
            temp = right
            right = min(right + cost[i - 1], left + cost[i - 2])
            left = temp

        return right

    def dominantIndex(self, nums: List[int]) -> int:
        # 维护一个二维列表即可,只需要扫描一遍，但是需要花费空间来存储两个值以及相应的元素
        lens, pos = len(nums), -1
        max_dig = [0] * 2
        for i in range(lens):
            if nums[i] > max_dig[0]:
                max_dig[1] = max_dig[0]
                max_dig[0] = nums[i]
                pos = i
            elif nums[i] > max_dig[1]:
                max_dig[1] = nums[i]

        return pos if max_dig[1] * 2 <= max_dig[0] else -1

    def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
        # 对于每个对角线作循环判断即可，从右向左，参数设置方面需要好好斟酌
        row, col = len(matrix), len(matrix[0])
        for i in range(col - 2, -row, -1):
            # 先整第一行的
            if i > -1:
                cur_row, cur_col = 1, i + 1
            # 再整第一列的
            else:
                cur_row, cur_col = -i, 1
            while cur_row < row and cur_col < col:
                if matrix[cur_row][cur_col] != matrix[cur_row - 1][cur_col - 1]:
                    return False
                cur_row += 1
                cur_col += 1
        return True

    def largeGroupPositions(self, S: str) -> List[List[int]]:
        lens, res, cur_ch, cur_pos = len(S), [], S[0], 0
        # 边缘判断一定不能少
        if lens < 3:
            return res
        for i in range(1, lens):
            if S[i] != cur_ch:
                if i - cur_pos >= 3:
                    res.append([cur_pos, i - 1])

                cur_ch = S[i]
                cur_pos = i

        # 注意最后的判断与之前的不一样
        if i - cur_pos + 1 >= 3:
            res.append([cur_pos, i])
        return res

    def flipAndInvertImage(self, A: List[List[int]]) -> List[List[int]]:
        # py大法好呀
        # 先反转，再与1异或即可
        return [list(map(lambda x: x ^ 1, rows[::-1])) for rows in A]

    def numMagicSquaresInside(self, grid: List[List[int]]) -> int:
        row, col = len(grid), len(grid[0])
        if row < 3 or col < 3:
            return 0

        # 判断幻方函数
        def check(mat: List[List[int]], st_row: int, st_cor: int) -> bool:
            lists = sum([mat[st_row + k][st_cor:st_cor + 3] for k in range(3)], [])
            if len(set(lists)) != 9 or max(lists) > 9 or min(lists) < 1:
                return False
            # ans = sum(mat[st_row][st_cor:st_cor+3])
            ans = 15
            # 其实还可以简化的，懒的搞了
            for i in range(1, 3):
                if sum(mat[i + st_row][st_cor:st_cor + 3]) != ans or sum(
                        mat[st_row + k][st_cor + i] for k in range(3)) != ans:
                    return False
                if i == 1:
                    if sum(mat[st_row + k][st_cor + k] for k in range(3)) != ans:
                        return False
                else:
                    if sum(mat[st_row + k][st_cor + 2 - k] for k in range(3)) != ans:
                        return False

            return True

        tot = 0
        for j in range(row - 2):
            for l in range(col - 2):
                if check(grid, j, l):
                    tot += 1

        return tot

    def maxDistToClosest(self, seats: List[int]) -> int:
        # 由于最开始与最后面的有区别，所以只能单独使用一个变量来保存当前最大的值
        cur_st, cur_ed, flag, max_st, max_ed, max_num, lens = 0, 0, True, 0, 0, 0, len(seats)
        for i in range(lens):
            if seats[i]:
                cur_ed = i
                # flag用于标记初始位置的特殊判断
                if flag:
                    max_num, max_st, max_ed = cur_ed - cur_st, cur_st, cur_ed
                    flag = False

                # 常规判断
                if (cur_ed - cur_st + 1) // 2 > max_num:
                    max_st, max_ed, max_num = cur_st, cur_ed, (cur_ed - cur_st + 1) // 2

                # 使用此处标记方便后续判断
                cur_st = i + 1

        # 用于最后一段特殊判断
        if cur_st < lens:
            cur_ed = lens - 1
            if cur_ed - cur_st + 1 > max_num:
                max_num = cur_ed - cur_st + 1

        return max_num

    def transpose(self, A: List[List[int]]) -> List[List[int]]:
        # 一种初始化多维列表的方法
        # r = [[0 for i in range(m)] for j in range(n)]
        # this will return a list[tuple()]
        # return list(zip(*A))
        return [list(row) for row in zip(*A)]

    def fairCandySwap(self, A: List[int], B: List[int]) -> List[int]:
        # flag用来判断选的是A还是B， leetcode编译器不通过顺序不对的列表，说好的山盟海誓嘞！
        sum_max, sum_min, flag = sum(A), sum(B), True
        chosse_sort, search_sort = A, set(B)
        if sum_max < sum_min:
            chosse_sort, search_sort, flag = B, set(A), False

        # 两者的差值
        sub_abs = abs(sum_max - sum_min) // 2
        # 使用set集合作为查找，减少查询时间
        for i in chosse_sort:
            if i - sub_abs in search_sort:
                return [i - sub_abs, i] if not flag else [i, i - sub_abs]

        return []

    def isMonotonic(self, A: List[int]) -> bool:
        # flag 标记递增递减序列
        lens, flag, st = len(A), True, 1
        pre = A[0]
        # 需要过滤与第一个相等的元素（1,1,0）
        while st < lens and A[st] == pre:
            st += 1
        for i in range(st, lens):
            # 从不重复的第一个元素开始判断
            if A[i] < pre and i == st:
                pre, flag = A[i], False
                continue

            # 递增或者递减判断
            if (flag and A[i] < pre) or (not flag and A[i] > pre):
                return False
            pre = A[i]

        return True

    def sortArrayByParity(self, A: List[int]) -> List[int]:
        # 直接给定两个列表存储即可
        sort_even, sort_odd = [], []
        for i in A:
            sort_odd.append(i) if i % 2 else sort_even.append(i)
        return sort_even + sort_odd

    def hasGroupsSizeX(self, deck: List[int]) -> bool:
        # 其实如果只是提取出现次数，使用列表添加也行
        dicts = {}
        for i in deck:
            if i in dicts:
                dicts[i] += 1
            else:
                dicts[i] = 1

        # set 缩小判断范围
        table = list(set(dicts.values()))
        x = table[0]

        def comdiv(div1: int, div2: int) -> int:
            # 辗转相除法，题解有错的呀
            while div2:
                temp = div1 % div2
                div1 = div2
                div2 = temp

            return div1

        for i in table[1:]:
            x = comdiv(x, i)
        return True if len(deck) > 1 and x > 1 else False

    def sortArrayByParityII(self, A: List[int]) -> List[int]:
        lens = len(A) // 2
        sort_even, sort_odd = [], []
        for i in A:
            sort_odd.append(i) if i % 2 else sort_even.append(i)

        # 或许还能简化
        res = []
        for i in range(lens):
            res.append(sort_even[i])
            res.append(sort_odd[i])
        return res

    def validMountainArray(self, A: List[int]) -> bool:
        lens = len(A)
        # 判别长度
        if lens < 3:
            return False

        max_val = max(A)
        max_pos = A.index(max_val)

        # 最大元素位置不能出现在首部以及尾部，并且不能出现多次，左右两边分别为顺序与逆序，并且不能含有重复元素
        if (max_pos != lens - 1 and max_pos and A[max_pos + 1] != max_val and sorted(set(A[:max_pos + 1])) == A[
                                                                                                              :max_pos + 1]
                and sorted(set(A[max_pos + 1:]), reverse=True) == A[max_pos + 1:]):
            return True
        return False

    def sortedSquares(self, A: List[int]) -> List[int]:
        return sorted(list(map(lambda x: x * x, A)))

    def sumEvenAfterQueries(self, A: List[int], queries: List[List[int]]) -> List[int]:
        sum_even = sum([x for x in A if not x % 2])
        lens = len(queries)
        ans = []
        for i in range(lens):
            cur = queries[i][0] + A[queries[i][1]]
            # 初始为偶，后为奇时要减去初始
            if cur % 2 and not A[queries[i][1]] % 2:
                sum_even -= A[queries[i][1]]
            # 初始为奇，后为偶时要添加当前值
            elif not cur % 2 and A[queries[i][1]] % 2:
                sum_even += cur
            # 初始为偶，后为偶时要添加增量值
            elif not cur % 2 and not A[queries[i][1]] % 2:
                sum_even += queries[i][0]

            # 列表也需要进行更新
            A[queries[i][1]] = cur
            ans.append(sum_even)

        return ans

    def addToArrayForm(self, A: List[int], K: int) -> List[int]:
        A, i = A[::-1], 0
        while K:
            # cur保存当前被加数
            cur, K = K % 10, K // 10
            # 注意A的长度是动态变化的
            cre, j, lens = 0, i + 1, len(A)
            # 当前数组长度较短时，直接添加
            if i == lens:
                A.append(cur)
            else:
                # 否则需要判断是否大于9
                A[i] = A[i] + cur
                if A[i] > 9:
                    A[i] -= 10
                    cre = 1
                while cre and j < lens:
                    A[j] += cre
                    if A[j] > 9:
                        A[j] = 0
                        cre = 1
                        j += 1
                    else:
                        cre = 0
                # 尾数判断
                if cre:
                    A.append(1)
            i += 1

        return A[::-1]

    def numRookCaptures(self, board: List[List[str]]) -> int:
        row, col, total = 0, 0, 0
        # 先找到R的位置
        for i in range(8):
            for j in range(8):
                if board[i][j] == 'R':
                    row, col = i, j

        pos_cre = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        # 再从四个方向分别判断即可
        for i in pos_cre:
            cur_row, cur_col = i[0] + row, i[1] + col
            while 0 <= cur_row < 8 and 0 <= cur_col < 8:
                if board[cur_row][cur_col] == 'B':
                    break
                elif board[cur_row][cur_col] == 'p':
                    total += 1
                    break
                cur_row += i[0]
                cur_col += i[1]

        return total

    def commonChars(self, A: List[str]) -> List[str]:
        # 此方法过于智障
        """
        lens = len(A)
        if lens < 2:
            return [x for x in A[0]]

        # 使用字典逐一存储，既耗空间又浪费时间
        all_dict, sort_ch = [], []
        for str_arr in A[1:]:
            dicts = {}
            for i in str_arr:
                if i not in dicts:
                    dicts[i] = 1
                else:
                    dicts[i] += 1
            all_dict.append(dicts)

        for ch in A[0]:
            count = 0
            for diction in all_dict:
                if ch in diction and diction[ch] > 0:
                    diction[ch] -= 1
                    count += 1
                elif ch not in diction or diction[ch] < 0:
                    break
            if count == lens - 1:
                sort_ch.append(ch)

        return sort_ch
        """
        # 统计每个字符出现次数的最小值即可,妙啊
        ans = []
        if not A:
            return []
        for ch in set(A[0]):
            # 统计结果
            count = [w.count(ch) for w in A]
            # 如果并没有全部出现，则count的最小值必为0
            s = ch * min(count)
            for i in s:
                ans.append(i)

        return ans

        # 可简写为一行解决
        # return [i for ch in set(A[0]) for i in ch * min(w.count(ch) for w in A)] if A else []

    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        # 暴力法，果不其然会超时
        """
        lens, totoal = len(time), 0
        for i in range(lens-1):
            for j in range(i+1, lens):
                if not (time[i]+time[j]) % 60:
                    totoal+= 1
        return totoal
        """
        # 预处理将数据区间缩小
        time = [x % 60 for x in time]

        from collections import defaultdict
        dicts = defaultdict(int)
        total = 0

        for t in time:
            # 处理余数为0时候的情况
            reside = (60 - t) % 60
            if reside in dicts:
                total += dicts[reside]

            # 注意字典中保存的是余数， 因此需要将其求余处理
            dicts[t] += 1

        return total
        # 在必须进行逐一查找时，务必考虑字典或者set，将查询时间大大缩短，以空间换时间

    def canThreePartsEqualSum(self, A: List[int]) -> bool:
        # 用两个指针判断位置即可， 最终结果已经给出，因此以是否相等为结束条件
        ans = sum(A)
        if ans % 3:
            return False
        ans = ans // 3
        st, ed, left_t, right_t = 0, len(A) - 1, A[0], A[-1]
        while st < ed and left_t != ans:
            st += 1
            left_t += A[st]

        while st < ed and right_t != ans:
            ed -= 1
            right_t += A[ed]

        if st < ed and left_t == right_t == sum(A[st + 1:ed]):
            return True
        return False

    def prefixesDivBy5(self, A: List[int]) -> List[bool]:
        # 顺序有错
        """
            A, lens, cur_val, ans = A[::-1], len(A), 0, []
            for i in range(lens-1, -1, -1):
                cur_val += 2**(lens-1-i)*A[i]
                if cur_val % 5:
                    ans.append(False)
                else:
                    ans.append(True)

            return ans
        """
        """
        # 强行使用内置方法过于耗时
        lens = len(A)
        ans, cur_str = [], ""
        for i in range(lens):
            cur_str += str(A[i])
            if not int(cur_str, 2) % 5:
                ans.append(True)
            else:
                ans.append(False)

        return ans
        """
        # 使用移位运算稍微优化了时间，但是还是不够
        """
        ans = []
        tot = int("".join(map(str, A[:])), 2)
        for _ in A:
            if tot % 5:
                ans.append(False)
            else:
                ans.append(True)
            tot = tot >> 1

        return ans[::-1]
        """
        # 大神思路：紧跟最后一位即可，无需全部计算，换位思考很重要
        ans, tail = [], 0
        for i in A:
            # 只需要记录尾数即可
            tail = tail * 2 + i
            if tail > 9:
                tail -= 10
            if not tail or tail == 5:
                ans.append(True)
            else:
                ans.append(False)

        return ans

    def countCharacters(self, words: List[str], chars: str) -> int:
        dicts, res = {}, 0

        # 统计字母表
        for i in chars:
            if i not in dicts:
                dicts[i] = 1
            else:
                dicts[i] += 1

        # 逐一判断词汇表
        for strs in words:
            # 不能直接使用dicts，每次判断都需要原来的数量
            cur_dict, count = dicts.copy(), 0
            for i in strs:
                # 小技巧：对于这种总量的判别，可以设一个变量统计总数，相等即满足条件，判断不满足逻辑太麻烦
                # 注意数量上的判断
                if i in cur_dict and cur_dict[i] > 0:
                    count += 1
                    cur_dict[i] -= 1

            if count == len(strs):
                res += count

        return res


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

    # 509 斐波那契数
    # print(show.fib(0))

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

    # 746 使用最小花费爬楼梯
    # print(show.minCostClimbingStairs([1, 100, 1, 1, 1, 100, 1, 1, 100, 1]))

    # 747 至少是其他数字两倍的最大数
    # print(show.dominantIndex([1,2,3,4]))

    # 766 托普利兹矩阵
    # print(show.isToeplitzMatrix([[41,45],[81,41],[73,81],[47,73],[0,47],[79,76]]))

    # 830 较大分组位置
    # print(show.largeGroupPositions("aaa"))

    # 832 翻转图像
    # print(show.flipAndInvertImage([[1,1,0,0],[1,0,0,1],[0,1,1,1],[1,0,1,0]]))

    # 840 矩阵中的幻方
    # print(show.numMagicSquaresInside([[4,3,8,4],[9,5,1,9],[2,7,6,2]]))

    # 849 到最近的人的最大距离
    # print(show.maxDistToClosest([0, 0, 0, 0, 1, 0, 0, 0]))

    # 867 转置矩阵
    # print(show.transpose([[1,2,3],[4,5,6],[7,8,9]]))

    # 888 公平的糖果交换
    # print(show.fairCandySwap([3,2],[1]))

    # 896 单调数列
    # print(show.isMonotonic([1,2,1,1,0]))

    # 905 按奇偶排序数组
    # print(show.sortArrayByParity([3,1,2,4]))

    # 914 卡牌分组
    # print(show.hasGroupsSizeX([1,1,1,2,2,2,3,3]))

    # 922 按奇偶排序数组II
    # print(show.sortArrayByParityII([4,2,5,7]))

    # 941 有效的山脉数组
    # print(show.validMountainArray([3,2,1]))

    # 977 有序数组的平方
    # print(show.sortedSquares([-7,-3,2,3,11]))

    # 985 查询后的偶数和
    # print(show.sumEvenAfterQueries([1,2,3,4],[[1,0],[-3,1],[-4,0],[2,3]]))

    # 989 数组形式的整数加法
    # print(show.addToArrayForm([0],100000))

    # 999 车的可用补货量
    # print(show.numRookCaptures([[".", ".", ".", ".", ".", ".", ".", "."], [".", ".", ".", "p", ".", ".", ".", "."],
    #                            [".", ".", ".", "R", ".", ".", ".", "p"], [".", ".", ".", ".", ".", ".", ".", "."],
    #                            [".", ".", ".", ".", ".", ".", ".", "."], [".", ".", ".", "p", ".", ".", ".", "."],
    #                            [".", ".", ".", ".", ".", ".", ".", "."], [".", ".", ".", ".", ".", ".", ".", "."]]))

    # 1002 查找常用字符
    # print(show.commonChars(["cool", "lock", "cook"]))

    # 1010 总持续时间可被60整除的歌曲
    # print(show.numPairsDivisibleBy60([30,20,150,100,40]))

    # 1013 将数组分成和相等的三个部分
    # print(show.canThreePartsEqualSum([18, 12, -18, 18, -19, -1, 10, 10]))

    # 1018 可被5整除的二进制前缀
    # print(show.prefixesDivBy5([1,1,0,0,0,1,0,0,1]))

    # 1160 拼写单词
    print(show.countCharacters(["cat","bt","hat","tree"], "atach"))
