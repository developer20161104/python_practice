from typing import List


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


# 380 常数时间插入删除与获取元素
class RandomizedSet:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.array = []
        self.dicts = {}
        # 长度是动态变化的，而元素位置是静态的有问题
        # self.lens = 0

    def insert(self, val: int) -> bool:
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        """
        if val not in self.dicts:
            self.array.append(val)
            # self.lens += 1
            self.dicts[val] = len(self.array) - 1
            return True

        return False

    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        """
        if val in self.dicts:
            index = self.dicts[val]
            # 此处交换有问题
            # self.array[index], self.array[-1] = self.array[-1], self.array[index]
            last_e = self.array[-1]
            # 一开始完全没想到的问题！！！1
            # 先将待删除元素与尾部元素交换，再将尾部元素下标修改
            self.array[index], self.dicts[last_e] = last_e, index

            self.array.pop()
            # 注意还要弹出dict中的元素
            del self.dicts[val]

            return True

        return False

    def getRandom(self) -> int:
        """
        Get a random element from the set.
        """
        import random
        return self.array[random.randint(0, len(self.array) - 1)]


# 729 我的日程安排表I
class MyCalendar:

    def __init__(self):
        self.time_arr = []

    def book(self, start: int, end: int) -> bool:
        # 使用二分法保持列表有序，由于日程无法重叠，因此最终结果以st或ed排序皆为相同
        # 以end进行排序
        left, right = 0, len(self.time_arr)
        # 通过二分查找找到待插入位置，判断标准为 end 与 mid 的 st
        # 找到右边界
        while left < right:
            mid = left + (right-left)//2
            if end > self.time_arr[mid][0]:
                left = mid+1
            else:
                right = mid

        # 判断标准为上一个日程的end与当前待插入的st
        # 如果st较小说明此日程与上一个日程重叠
        if left > 0 and self.time_arr[left-1][1] > start:
            return False
        else:
            # 否则直接插入位置即可
            self.time_arr.insert(left, [start, end])
            return True


class Solution:
    def __init__(self):
        self.order = 0

        # 611
        # self.count = 0

        # 695
        # self.max_area = 0
        # self.cur_area = 1

    def maxArea(self, height: List[int]) -> int:
        # 转化为求取最大面积，长度可变，宽度为两边的min
        left, right = 0, len(height) - 1
        max_area = 0
        while left < right:
            # 保存最佳状态
            max_area = max(max_area, min(height[left], height[right]) * (right - left))

            # 向内收缩只可能在宽高增大的情况下面积变大
            if height[left] > height[right]:
                right -= 1
            else:
                left += 1

        return max_area

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # 超时可还行，主要是每步都需要从前向后遍历，需要剪枝
        # sort_arr = sorted(nums)
        # lens = len(nums)
        # left, right = 0, lens - 1
        #
        # res = []
        # while left < right:
        #     cur_sum = sort_arr[left] + sort_arr[right]
        #     l, r = left + 1, right - 1
        # 内部使用二分思路是没错
        #     while l <= r:
        #         mid = l + (r - l) // 2
        #         cur_find = sort_arr[mid]
        #         if not cur_find + cur_sum and [sort_arr[left], sort_arr[right], sort_arr[mid]] not in res:
        #             res.append([sort_arr[left], sort_arr[right], sort_arr[mid]])
        #         elif cur_find + cur_sum < 0:
        #             l = mid + 1
        #         else:
        #             r = mid - 1
        #
        #     if sort_arr[right-1] + cur_sum < 0:
        #         left += 1
        #         right = lens-1
        #     else:
        #         right -= 1
        #
        # return list(res)

        # 大神解法
        # 不使用二分查找，逐一进行比对
        # 左边低值右边高值，以及中间的值进行寻找
        lens = len(nums)
        res = []
        nums.sort()

        if lens < 3:
            return res

        # 固定最左边的值，从后续元素中查找（左小右大）
        for i in range(lens):
            # 剪枝一：如果左边已经大于0，最终结果不可能为0
            if nums[i] > 0:
                return res

            # 过滤重复
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            # 初值设置
            L = i + 1
            R = lens - 1
            # 每次查找的思路不同：
            while L < R:
                if nums[i] + nums[L] + nums[R] == 0:
                    res.append([nums[i], nums[L], nums[R]])

                    # 内部虑重
                    while L < R and nums[L] == nums[L + 1]:
                        L += 1
                    while L < R and nums[R] == nums[R - 1]:
                        R -= 1
                    # 需要两边都增加，过滤掉当前满足的两个值
                    L += 1
                    R -= 1

                # 根据求解来移动指针
                elif nums[i] + nums[L] + nums[R] > 0:
                    R -= 1
                else:
                    L += 1

        return res

    def threeSumClosest(self, nums: List[int], target: int) -> int:
        # 跟三数和的思想很像，但是更简单，不需要滤重
        lens = len(nums)
        nums.sort()

        res = sum(nums[:3])

        # 设置首位以及双指针，分别表示三个值
        for i in range(lens - 2):
            left, right = i + 1, lens - 1

            while left < right:
                cur_res = nums[i] + nums[left] + nums[right]
                # 注意比较的是绝对值
                if abs(res - target) > abs(cur_res - target):
                    res = cur_res
                if cur_res - target == 0:
                    return res
                elif cur_res - target < 0:
                    left += 1
                else:
                    right -= 1

        return res

    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        # 使用双重循环来拓展三数和
        nums.sort()
        res, lens = [], len(nums)

        for i in range(lens - 3):
            # 注意此处也需要滤重
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            # 这个剪枝没想到
            # 当前最小的部分进行判别
            if nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target:
                break
            # 当前最大的部分进行判别
            if nums[i] + nums[lens - 1] + nums[lens - 2] + nums[lens - 3] < target:
                continue

            for j in range(i + 1, lens - 2):

                # 过滤条件：参照三数和
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                # 当前最小判断
                if nums[i] + nums[j] + nums[j + 2] + nums[j + 1] > target:
                    break
                # 当前最大判断
                if nums[i] + nums[j] + nums[lens - 1] + nums[lens - 2] < target:
                    continue

                left, right = j + 1, lens - 1

                # 双指针法压缩时间
                while left < right:
                    cur_sum = nums[i] + nums[j] + nums[left] + nums[right]
                    if cur_sum == target:
                        res.append([nums[i], nums[j], nums[left], nums[right]])
                        while left < right and nums[left + 1] == nums[left]:
                            left += 1
                        while left < right and nums[right - 1] == nums[right]:
                            right -= 1

                        left += 1
                        right -= 1
                    elif cur_sum < target:
                        left += 1
                    else:
                        right -= 1

        return res

    def nextPermutation(self, nums: List[int]) -> None:
        # 经典字典序法
        i, j, k = 0, 0, 0
        lens = len(nums)
        # 边界判断
        if lens < 2:
            return

        # 从右到左寻找顺序下标
        flag = True
        for i in range(lens - 1, 0, -1):
            if nums[i] > nums[i - 1]:
                j, flag = i - 1, False
                break

        # 如果一次都没出现过表示已经是最终序
        if flag:
            nums.sort()
            return

        k = j + 1
        for i in range(j + 2, lens):
            # 注意在出现相同的字符时，需要选择最后一个字符
            if nums[j] < nums[i] <= nums[k]:
                k = i

        # 交换位置
        nums[j], nums[k] = nums[k], nums[j]
        # 翻转j+1后的字符，完成一次字典序
        for i in range(1, (lens - j - 1) // 2 + 1):
            nums[j + i], nums[lens - i] = nums[lens - i], nums[i + j]

        return

    def search(self, nums: List[int], target: int) -> int:
        # error: 案例[4,5,6,7,8,1,2,3], target=8
        # left, right = 0, len(nums)-1
        #
        # # 错误二：边界判断
        # if not nums:
        #     return -1
        #
        # while left < right:
        #     mid = left + (right-left)//2
        #
        #     if nums[mid] == target:
        #         return mid
        #     else:
        #         if target < min(nums[left], nums[right]):
        #             left = mid+1
        #         elif target > max(nums[left], nums[right]):
        #             right = mid-1
        #         else:
        #             # 中间逻辑问题
        #             # if nums[left] != target and nums[right] != target:
        #             #     return -1
        #             # return left if nums[left] == target else right
        #             if nums[left] != target and nums[right] != target:
        #                 if nums[mid] > target:
        #                     if nums[right] < nums[left]:
        #                         left = mid+1
        #                     else:
        #                         right = mid-1
        #                 elif nums[mid] < target:
        #                     if nums[right] < nums[left]:
        #                         right = mid-1
        #                     else:
        #                         left = mid +1
        #                 else:
        #                     return mid
        #             else:
        #                 return left if nums[left] == target else right
        #
        # # 错误一：单点判断
        # return left if nums[left] == target else -1

        # 把握关键点：如果一边无序，则另一边必为有序，而且是升序
        # 理解有问题，注意是围着某个点旋转，而不是部分或整体转置
        # 取左闭右开 right = len(nums)
        left, right = 0, len(nums) - 1

        # 由于需要进行压缩，所以必须判断左边与右边相等的情况
        # 更通用写法： left < right
        while left <= right:
            mid = left + (right - left) // 2

            if nums[mid] == target:
                return mid
            # 左边首尾是升序时
            elif nums[left] <= nums[mid]:
                # 当出现相等时不断压缩
                # 寻找较优判断条件，其余丢到else里面即可
                # 由于需要压缩左右空间，因此在相等时依然进行压缩，直到返回结果
                if nums[left] <= target <= nums[mid]:
                    # right = mid
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                # 否则就是右边升序！（围点旋转）
                # 由于右指针需要判断，因此此处的判断应该为 right-1
                if nums[mid] <= target <= nums[right]:
                    left = mid + 1
                else:
                    # right = mid
                    right = mid - 1

        return -1

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        # 最基本的二分加上线性搜索
        # left, right = 0, len(nums)
        #
        # if not nums:
        #     return [-1]*2
        #
        # while left < right:
        #     mid = left + (right-left)//2
        #     if nums[mid] == target:
        #         l, r = mid, mid
        #         # 在添加了线性搜索后，时间复杂度会退化为O(n)
        #         while l > -1 and nums[l] == target:
        #             l -= 1
        #         while r < len(nums) and nums[r] == target:
        #             r += 1
        #         return [l+1, r-1]
        #
        #     elif nums[mid] < target:
        #         left = mid+1
        #     else:
        #         right = mid
        #
        # return [-1]*2

        # 方法二：查找边界的二分
        def binary_search_left(nums: List[int], target: int) -> int:
            left, right = 0, len(nums)

            while left < right:
                mid = left + (right - left) // 2

                if nums[mid] < target:
                    left = mid + 1
                else:
                    # key：当找到元素位置时，不立即返回，而是继续收缩右边区间，
                    # 直到找到最左边目标值
                    right = mid

            # 由于当查找不到元素时，会返回插入下标，因此需要进行范围判断
            return left if len(nums) > left > -1 and nums[left] == target else -1

        def binary_search_right(nums: List[int], target: int) -> int:
            left, right = 0, len(nums)

            while left < right:
                mid = left + (right - left) // 2

                if nums[mid] > target:
                    right = mid
                else:
                    # key：当找到元素位置时，不立即返回，而是继续收缩左侧区间，
                    # 直到找到最右边目标值
                    left = mid + 1

            # 还是需要进行边界判断
            return left - 1 if len(nums) >= left > 0 and nums[left - 1] == target else -1

        if not nums:
            return [-1] * 2

        # 关键点：在找到目标值后，并不急于返回，而是压缩左（右）区间来寻找右（左）边界
        return [binary_search_left(nums, target), binary_search_right(nums, target)]

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        # error:仅能处理两个数组合
        # if not candidates:
        #     return []
        #
        # res = set()
        # value_set = set(candidates)
        # for n in candidates:
        #     cur_mod = target % n
        #     while cur_mod <= target:
        #         if cur_mod in value_set:
        #             res.add(tuple(sorted([n] * ((target-cur_mod) // n) + [cur_mod])))
        #         cur_mod += n
        #
        # return list(res)

        # 需要用到回溯加上DFS（超出当前能力范围）
        if not candidates:
            return []

        # 排序可以去掉重复路径
        candidates.sort()
        cur_path = []
        res = []

        self._dfs(candidates, target, 0, cur_path, res)

        return res

    def _dfs(self, candidates: List[int], target: int, begin: int, path: List[int], res: List[int]):
        # python 传参如果是可更改变量（list,dict），则相当于引用
        # 在内部函数修改后，会直接改变原始变量
        # 不可变变量有（number，tuple，string）
        if target == 0:
            # 由于总是更改原始变量，因此需要复制保留当前的可变变量（path）
            # 所以需要使用path[:] 或者 path.copy() 来进行处理
            res.append(path[:])
            return

        lens = len(candidates)
        for i in range(begin, lens):

            # 剪枝：如果当前目标值已经过小
            # 则直接结束回溯
            if target - candidates[i] < 0:
                return

            path.append(candidates[i])
            # 通过i来表示可重复，设置搜索起点，去除重复项，并缩小范围向下递归寻找
            self._dfs(candidates, target - candidates[i], i, path, res)

            # 回溯：弹出当前的值
            path.pop()

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        # 基本思想与上一题一致，但死活想不出过滤条件。。
        if not candidates:
            return []

        candidates.sort()
        path = []
        res = []
        self._dfs2(candidates, target, path, res, 0)
        return list(res)

    def _dfs2(self, candidates: List[int], target: int, path: List[int], res: List[int], start: int):
        if target == 0:
            res.append(path[:])
            return

        for i in range(start, len(candidates)):
            if target - candidates[i] < 0:
                return

            # 过滤不完全：一直以整体进行考虑
            # # 过滤了起始相同项
            # if not path and candidates[i] == pre[0]:
            #     continue

            # 放到部分中进行考虑：当前的值如果在首位之后并且又跟首位相同时，必会重复
            # 此时后面的序列遍历前面序列的子集合，因此可以删去
            if i > start and candidates[i - 1] == candidates[i]:
                continue

            path.append(candidates[i])

            # 在进行递归的时候以下一个点为起始值
            self._dfs2(candidates, target - candidates[i], path, res, i + 1)
            # 回溯模板
            path.pop()

    def rotate(self, matrix: List[List[int]]) -> None:
        # 整体思路没问题，就是多了过多中间步骤
        # lens = len(matrix)
        # if lens < 2:
        #     return
        #
        # times = lens // 2
        # move = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        # # 负责轮次
        # for i in range(times):
        #     cur_pos = [[i, i], [i, lens - 1 - i], [lens - i - 1, lens - i - 1], [lens - i - 1, i]]
        #     # 旋转次数为n-1
        #     for k in range(i, lens - i - 1):
        #         # 通过循环进行数值传输,没必要存储
        #         cur_val = [matrix[x[0]][x[1]] for x in cur_pos]
        #         for j in range(1, 5):
        #             matrix[cur_pos[j % 4][0]][cur_pos[j % 4][1]] = cur_val[j - 1]
        #
        #         cur_pos = list(map(lambda x, y: [x[0] + y[0], x[1] + y[1]], cur_pos, move))
        #
        # return

        # 考虑矩阵基本操作：转置与翻转
        lens = len(matrix)

        # 转置操作
        for i in range(lens):
            for j in range(i, lens):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

        # 翻转操作
        for i in range(lens):
            # 错误之处：转回来了
            for j in range(lens // 2):
                matrix[i][j], matrix[i][lens - 1 - j] = matrix[i][lens - 1 - j], matrix[i][j]

        print(matrix)
        return

    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        # 图遍历常用技巧：设置位移数组
        move = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        res = []
        flag = -10
        if not matrix:
            return res

        len_m, len_n = len(matrix), len(matrix[0])

        row, col, i = 0, 0, 0
        # 可将条件判断优化为无需最后一次判断，因为已经知道了需要进行多少次遍历
        # while matrix[row][col] != flag:
        for _ in range(len_m * len_n):
            res.append(matrix[row][col])
            # 设置标志位防止重复遍历
            matrix[row][col] = flag

            if not (0 <= row + move[i][0] < len_m and 0 <= col + move[i][1] < len_n) \
                    or matrix[row + move[i][0]][col + move[i][1]] == flag:
                i += 1

            # 注意方向调转时的下标
            i = i % 4
            row, col = row + move[i][0], col + move[i][1]
            # if not (0 <= row < len_m and 0 <= col < len_n):
            #     break

        return res

    def canJump(self, nums: List[int]) -> bool:
        # 贪心算法，有很多小细节
        lens = len(nums)
        if lens < 2:
            return True
        # 使用一个变量记录当前能到达的最大距离即可
        cur_max = -1

        for i in range(lens):
            # 出现0时进行判断
            if not nums[i]:
                # 能够跳到最后一个位置即为成功
                if cur_max <= i != lens - 1:
                    return False
                continue

            cur_max = max(cur_max, i + nums[i])

        return True

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if len(intervals) < 2:
            return intervals
        res = []
        # 排序了以后会减少很多麻烦
        intervals.sort()

        cur_min, cur_max = intervals[0][0], intervals[0][1]
        for cur in intervals[1:]:
            # 如果当前出现的下界比原来的上界还大，则需要重新开辟新区间
            if cur[0] > cur_max:
                res.append([cur_min, cur_max])
                cur_min = cur[0]
            # 错误一：不考虑大区间并子区间不用更改两边的情况
            # 错误案例：[[1,4],[2,3]]
            cur_max = max(cur_max, cur[1])

        res.append([cur_min, cur_max])
        return res

    def generateMatrix(self, n: int) -> List[List[int]]:
        # 使用列表推导来生成
        matrix = [[0] * n for _ in range(n)]
        move_row = [0, 1, 0, -1]
        move_col = [1, 0, -1, 0]
        pos = 0
        cur_row, cur_col = 0, 0
        # 整体思路是先构建空矩阵，然后按照要求逐一填充即可
        for i in range(n * n):
            matrix[cur_row][cur_col] = i + 1

            temp_row, temp_col = cur_row + move_row[pos], cur_col + move_col[pos]
            if not (0 <= temp_row < n and 0 <= temp_col < n) \
                    or matrix[temp_row][temp_col]:
                # 郁闷得一批，余数搞错了可还行
                pos = (pos + 1) % 4
                cur_row += move_row[pos]
                cur_col += move_col[pos]
            else:
                cur_row, cur_col = temp_row, temp_col

        return matrix

    def uniquePaths(self, m: int, n: int) -> int:
        # 动态规划来解决
        # 还能优化空间复杂度
        # mat = [[1]*n] + [[0] * n for _ in range(m-1)]
        #
        # for i in range(1, m):
        #     for j in range(n):
        #         # 边界条件都为1
        #         if not j:
        #             mat[i][j] = 1
        #         # 中间部分的需要加上两种路径的数量
        #         else:
        #             mat[i][j] = mat[i - 1][j] + mat[i][j - 1]
        #
        # # 最后返回最终结果
        # return mat[m - 1][n - 1]

        # 由于当前行仅需要上一行与当前行的前一个值，因此用两个列表保存即可
        # 空间优化为2*n
        # pre, cur = [1]*n, [1]*n
        # for i in range(1, m):
        #     for j in range(1, n):
        #         # 优化空间
        #         cur[j] = pre[j] + cur[j-1]
        #
        #     # 保存上一行的结果
        #     pre = cur[:]
        #
        # return cur[-1]

        # 由于当前列表中已经保存了上一次的所有结果
        # 因此无需再构建一个列表
        # 优化空间复杂度为 n
        cur = [1] * n

        # 每次递进一行
        for i in range(1, m):
            # 每次计算得到当前列的结果
            for j in range(1, n):
                cur[j] += cur[j - 1]

        return cur[-1]

    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        # if not obstacleGrid:
        #     return 0
        # len_m, len_n = len(obstacleGrid), len(obstacleGrid[0])
        #
        # # 寻找第一行中的障碍
        # cur = []
        # for i in range(len_n):
        #     if obstacleGrid[0][i]:
        #         break
        #     cur.append(1)
        # cur += [0]*(len_n-len(cur))
        #
        # # 寻找第一列中的障碍
        # row_pos = len_m + 1
        # for i in range(len_m):
        #     if obstacleGrid[i][0]:
        #         row_pos = i
        #         break
        #
        # for i in range(1, len_m):
        #     for j in range(1, len_n):
        #         if obstacleGrid[i][j]:
        #             cur[j] = 0
        #         # row_pos仅用于第一个判断
        #         elif j > 1 or i < row_pos:
        #             cur[j] += cur[j-1]
        #
        # # 如果出现障碍并且列宽为1，则必为0
        # return 0 if row_pos < len_m and len_n == 1 else cur[-1]

        # 可优化空间至1，使用矩阵来保存当前的规划值
        if not obstacleGrid or obstacleGrid[0][0]:
            return 0

        len_m, len_n = len(obstacleGrid), len(obstacleGrid[0])
        obstacleGrid[0][0] = 1

        # travel row
        for i in range(1, len_m):
            # 巧妙利用与或条件进行判断
            obstacleGrid[i][0] = int(obstacleGrid[i - 1][0] == 1 and obstacleGrid[i][0] == 0)

        # travel column
        for j in range(1, len_n):
            obstacleGrid[0][j] = int(obstacleGrid[0][j] == 0 and obstacleGrid[0][j - 1] == 1)

        # 在前面已经进行了行列遍历后，只需要逐步相加即可得到结果
        # 相比自己创造单一变量来判断更好理解
        for i in range(1, len_m):
            for j in range(1, len_n):
                if obstacleGrid[i][j]:
                    obstacleGrid[i][j] = 0
                else:
                    obstacleGrid[i][j] = obstacleGrid[i - 1][j] + obstacleGrid[i][j - 1]

        return obstacleGrid[len_m - 1][len_n - 1]

    def minPathSum(self, grid: List[List[int]]) -> int:
        if not grid:
            return 0

        len_m, len_n = len(grid), len(grid[0])

        # 首行路径求和
        for i in range(1, len_m):
            grid[i][0] += grid[i - 1][0]

        # 首列路径求和
        for j in range(1, len_n):
            grid[0][j] += grid[0][j - 1]

        # 选取两者的最短路径
        for i in range(1, len_m):
            for j in range(1, len_n):
                grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])

        return grid[len_m - 1][len_n - 1]

    def setZeroes(self, matrix: List[List[int]]) -> None:
        # # 先尝试m+n额外空间
        # if not matrix:
        #     return
        # len_m, len_n = len(matrix), len(matrix[0])
        # row_pos, col_pos = [0] * len_m, [0] * len_n
        #
        # # 先找出零元素行列坐标
        # for i in range(len_m):
        #     for j in range(len_n):
        #         if not matrix[i][j]:
        #             row_pos[i], col_pos[j] = 1, 1
        #
        # # 对行进行置零
        # for index, pos in enumerate(row_pos):
        #     if pos:
        #         matrix[index][:] = [0] * len_n
        #
        # # 对列进行置零
        # for index, pos in enumerate(col_pos):
        #     if pos:
        #         for i in range(len_m):
        #             matrix[i][index] = 0
        #
        # print(matrix)
        # return

        # 尝试常数空间算法
        # 利用列表自身来进行存储
        if not matrix:
            return
        len_m, len_n = len(matrix), len(matrix[0])

        # 两个标记用于判别第一行第一列是否置零
        row_flag, col_flag = False, False

        # 将每个零元位置的起始行位置，起始列位置置零，用于标记
        for i in range(len_m):
            for j in range(len_n):
                if not matrix[i][j]:
                    if not i:
                        row_flag = True
                    if not j:
                        col_flag = True
                    matrix[0][j], matrix[i][0] = 0, 0

        # 先将非第一行，第一列元素置零
        for i in range(1, len_m):
            for j in range(1, len_n):
                if not matrix[i][0] or not matrix[0][j]:
                    matrix[i][j] = 0

        # 最后再置零第一行第一列
        if row_flag:
            matrix[0][:] = [0] * len_n
        if col_flag:
            for i in range(len_m):
                matrix[i][0] = 0

        print(matrix)

    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix:
            return False

        len_m, len_n = len(matrix), len(matrix[0])
        row_pos = 0
        for i in range(1, len_m):
            # 边界取值有问题:不能用大于进行推断
            # 在边界会有各种情况
            if matrix[i][0] <= target:
                row_pos = i
            else:
                break

        # 判断是否在矩阵中
        if row_pos < 0:
            return False

        # 二分法判断具体位置
        left, right = 0, len_n
        while left < right:
            mid = left + (right - left) // 2
            if matrix[row_pos][mid] == target:
                return True
            elif target > matrix[row_pos][mid]:
                left = mid + 1
            else:
                right = mid

        return True if 0 <= left < len_n and matrix[row_pos][left] == target else False

        # 也可以直接使用二分法，然后对下标进行转化即可

    def sortColors(self, nums: List[int]) -> None:
        # 使用计数排序，消耗o(3)空间
        # counts = [0]*3
        # for num in nums:
        #     counts[num] += 1
        #
        # nums = []
        # for index, count in enumerate(counts):
        #     nums += [index]*count
        #
        # print(nums)

        # 三指针法，定义两个边界来处理
        # 在指针前进上遇到问题，还需要好好想想
        if not nums:
            return
        cur, left, right = 0, 0, len(nums) - 1
        # 类似于二分法，注意指针的判断
        while cur <= right:
            if not nums[cur]:
                # 注意需要交换
                nums[cur], nums[left] = nums[left], nums[cur]
                # 指针的移动需要仔细斟酌
                # 左边必定全都扫描过
                left += 1
                cur += 1
            elif nums[cur] == 1:
                # 位于中间就不用管
                cur += 1
            else:
                nums[cur], nums[right] = nums[right], nums[cur]
                # 此处仅需移动右指针，因为不知道右指针上个指向的元素大小
                right -= 1

        print(nums)

    def subsets(self, nums: List[int]) -> List[List[int]]:
        # error:还是没有领悟回溯的精髓
        # 关键错误：递归的下标选择有问题
        # res = [[]]
        # if not nums:
        #     return res
        #
        # nums.sort()
        # path = []
        # visit = [0]*len(nums)
        #
        # def findall(index: int):
        #     if index == len(nums) or visit[index]:
        #         return
        #
        #     for i in range(index, len(nums)):
        #         if not visit[i]:
        #             visit[i] = 1
        #             path.append(nums[i])
        #
        #             res.append(path[:])
        #             findall(index+1)
        #             path.pop()
        #             visit[i] = 0
        #
        # findall(0)
        # return res

        # 方法一：回溯
        # if not nums:
        #     return [[]]
        #
        # res = []
        # lens = len(nums)
        #
        # def backtrace(index, path):
        #     if len(path) == k:
        #         res.append(path[:])
        #
        #     else:
        #         for i in range(index, lens):
        #             # 内部中的下标选择的是当前下标
        #             # 不是index！！
        #             path.append(nums[i])
        #             # 回溯的下标是i，不是index
        #             # 被这里坑了1小时，还是不扎实
        #             backtrace(i + 1, path)
        #             path.pop()
        #
        # # 终止条件：长度从1次选到lens即可
        # for k in range(lens+1):
        #     backtrace(0, [])
        #
        # return res

        # 方法二：大神思路
        # 每次遍历一个元素，就将此元素添加到所有子集元素中
        import copy
        res = [[]]
        if not nums:
            return res

        lens = len(nums)
        res.append([nums[0]])
        for i in range(1, lens):
            # 此处需要用到深拷贝
            # 由于此处的复制为嵌套列表（高维），因此需要深拷贝
            # 如果待拷贝列表内部为普通类型时，深浅拷贝几乎一致
            cur = copy.deepcopy(res)
            for j in range(len(res)):
                res[j].append(nums[i])
            res += cur

        return res

    def exist(self, board: List[List[str]], word: str) -> bool:
        # 一个经典的二维平面
        len_row, len_col = len(board), len(board[0])
        # 位置移动
        move_row, move_col = [0, -1, 0, 1], [-1, 0, 1, 0]
        len_t = len(word)
        # 访问数组
        visit = [[0] * len_col for _ in range(len_row)]

        def travel_arround(cur_row: int, cur_col: int, word_index: int):
            # 截止条件是访问完成
            if word_index == len_t:
                return True

            for k in range(4):
                temp_row, temp_col = cur_row + move_row[k], cur_col + move_col[k]
                # BFS遍历
                if 0 <= temp_row < len_row and 0 <= temp_col < len_col and \
                        board[temp_row][temp_col] == word[word_index] and not visit[temp_row][temp_col]:
                    visit[temp_row][temp_col] = 1
                    res = travel_arround(temp_row, temp_col, word_index + 1)
                    # 回溯
                    visit[temp_row][temp_col] = 0
                    # 一旦出现满足的情况，则立即返回结果
                    if res:
                        return True

            return False

        for i in range(len_row):
            for j in range(len_col):
                if board[i][j] == word[0]:
                    # 外部回溯
                    visit[i][j] = 1
                    if travel_arround(i, j, 1):
                        return True
                    visit[i][j] = 0

        return False

    def removeDuplicates(self, nums: List[int]) -> int:
        # 此方式无法保证有序
        # if not nums:
        #     return 0
        # lens = len(nums)
        # pos, cur, time = lens - 1, nums[0], 0
        #
        # for i in range(lens):
        #     if nums[i] == cur:
        #         time += 1
        #     else:
        #         if time > 2:
        #             for j in range(time - 2):
        #                 nums[i - j -1], nums[pos - j] = nums[pos - j], nums[i -j-1]
        #             pos -= time - 2
        #
        #         cur, time = nums[i], 1
        #
        # print(nums)
        # return pos+1

        # 时间复杂度为O(n**2)，可以优化
        # if not nums:
        #     return 0
        # # 添加末尾判断符号
        # nums.append(-1)
        #
        # lens = len(nums)
        # cur, time = nums[0], 0
        # count, i = lens, 0
        #
        # # 问题1，末尾判断
        # while i < count:
        #     if cur == nums[i]:
        #         time += 1
        #         i += 1
        #     else:
        #         if time > 2:
        #             # 将有效长度全体左移，过于耗时
        #             for j in range(i, count):
        #                 nums[j-time+2] = nums[j]
        #             count -= time - 2
        #             i = i-time+2
        #         cur, time = nums[i], 0
        #
        # print(nums)
        # return count-1

        # 大神思路：快慢指针 O(n)
        # 思想是逐一移动，就不需要多余的重复移动次数
        lens = len(nums)
        if lens < 3:
            return lens

        # 使用j作为慢指针
        time, j, cur = 0, 0, nums[0]
        for i in range(lens):
            if nums[i] == cur:
                time += 1
            else:
                # 新元素重新计数
                cur, time = nums[i], 1

            # 当前仅当当前元素出现次数少于3时，慢指针才会递增
            if time < 3:
                nums[j] = nums[i]
                j += 1

        print(nums)
        return j

    def search(self, nums: List[int], target: int) -> bool:
        # 经典案例 ：1 3 1 1 与 1 1 3 1
        left, right = 0, len(nums)
        # 前面是二分的改版
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return True
            # 先判断有序的一方
            elif nums[left] < nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid
                else:
                    left = mid + 1
            # 由于两边可能相同，因此需要单独进行考虑
            elif nums[mid] < nums[right - 1]:
                if nums[mid] < target <= nums[right - 1]:
                    left = mid + 1
                else:
                    right = mid
            # 特别针对两边相等的情况
            else:
                # 后面其实只需要优化为左指针右移一位即可（有些冗余）
                # 如果出现重复元素，则退化为逐步搜索
                if nums[left] == target:
                    return True
                i = left + 1
                while i < mid:
                    if nums[i] != nums[left]:
                        break
                    i += 1

                # 典型的打补丁式算法
                if i == mid:
                    left = mid + 1
                else:
                    right = mid

        return False if not nums or not (0 <= left < len(nums)) or \
                        nums[left] != target else True

    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        if not nums:
            return [[]]

        lens = len(nums)
        res = []
        cur_count = []
        nums.sort()

        def get_all(index=0):
            if len(cur_count) == k:
                res.append(cur_count[:])
                return

            for i in range(index, lens):
                # 在剪枝上还是有小问题
                # 此处选为index是为了让每次的起始部分能够顺利加入集合中
                # 需要排除的是同一层级上的多余部分，
                # 对于不同层级不需要排除（因此不能使用i>0来进行判断）
                if i > index and nums[i] == nums[i - 1]:
                    continue
                # if i > 0 and not len(cur_count) and nums[i] == nums[i-1]:
                #     continue

                cur_count.append(nums[i])
                get_all(i + 1)
                # 回溯
                cur_count.pop()

        # 长度是个变数
        for k in range(lens + 1):
            get_all()

        return res

    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        lens = len(preorder)
        if not lens:
            return None

        # 错误一：下标指向问题
        # 错误二：局部变量引起的下标问题
        # 每次使用后必须将下标增加，
        # 使用局部变量会在栈保存的时候保留原始值，造成下标问题
        # 有分治法的思想
        def build(left: int, right: int) -> TreeNode:
            T = TreeNode(None)

            if self.order < lens:
                # 如果当前值存在则新建节点
                T.val = preorder[self.order]
                # 利用先序与中序的特点进行构造
                cur_mid = inorder.index(preorder[self.order])
                # 注意先增加下标再进行添加
                if left < cur_mid:
                    self.order += 1
                    T.left = build(left, cur_mid - 1)
                if right > cur_mid:
                    # 如果没有左指针，直接加2会溢出
                    # T.right = build(order + 2, cur_mid + 1, right)
                    self.order += 1
                    T.right = build(cur_mid + 1, right)
            return T

        return build(0, lens - 1)

    def travel_tree_bfs(self, T: TreeNode):
        from collections import deque
        q = deque()
        if T:
            q.append(T)
        level = 0
        while q:
            lens = len(q)
            level += 1
            # 弹出位置搞错了，尴尬
            for i in range(lens):
                cur_T = q.popleft()
                print('level ', level, ': ', cur_T.val)
                if cur_T:
                    if cur_T.left:
                        q.append(cur_T.left)
                    if cur_T.right:
                        q.append(cur_T.right)

    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        if not inorder or not postorder:
            return None

        lens = len(inorder)
        self.order = lens - 1

        # 中序与后序建立二叉树的话需要先创建右子树再生成左子树
        # 其他的内容与前序中序生成一样
        def create_tree(left: int, right: int):
            T = TreeNode(None)
            if self.order >= 0:
                T.val = postorder[self.order]
                cur_mid = inorder.index(postorder[self.order])
                # 先右后左
                if right > cur_mid:
                    self.order -= 1
                    T.right = create_tree(cur_mid + 1, right)
                if left < cur_mid:
                    self.order -= 1
                    T.left = create_tree(left, cur_mid - 1)

            return T

        return create_tree(0, lens - 1)

    def minimumTotal(self, triangle: List[List[int]]) -> int:
        # 直接在给定列表中操作，算不算是O(1)/doge
        if not triangle:
            return 0

        lens = len(triangle)
        for i in range(1, lens):
            len_cur = len(triangle[i])

            # 处理边界情况
            triangle[i][0] += triangle[i - 1][0]
            triangle[i][-1] += triangle[i - 1][-1]

            # 动态规划思想
            for j in range(1, len_cur - 1):
                triangle[i][j] += min(triangle[i - 1][j], triangle[i - 1][j - 1])

        return min(triangle[-1])

    def maxProduct(self, nums: List[int]) -> int:
        # 保留状态后续处理的思想完全不行

        # if not nums:
        #     return 0
        # cur_res = nums[0]
        # state = []
        # nums.append(0)
        # res = nums[0]
        #
        # for num in nums[1:]:
        #     if num > 0:
        #         if not cur_res:
        #             cur_res = num
        #         else:
        #             cur_res *= num
        #     elif num < 0:
        #         state.append([cur_res, num])
        #         cur_res = 0
        #     else:
        #         len_st = len(state)
        #         if len_st > 1:
        #             res = state[0][0]*state[0][1]
        #             for d in state[1:len_st-len_st%2]:
        #                 res *= d[0]*d[1]
        #         else:
        #             # 只出现一次的时候进行此处判断
        #             if state:
        #                 res = max(res, state[0][0])
        #
        #         res = max(res, cur_res)
        #         # 遇到0需要清空所有状态
        #         state = []
        #
        # return res

        # 死都想不出来
        # 关键点：由于负数的存在，会造成最大值变成最小，最小值变成最大，因此还需要保存当前最小值
        # 典型的动态规划，但是是双层的
        max_d = -20000
        # imax为以当前点作为最后一个点时的最大值
        # imin为以当前点作为最后一个点时的最小值
        imax, imin = 1, 1
        for num in nums:
            if num < 0:
                imax, imin = imin, imax

            # 此处有点类似于最大子序列的递推
            imax = max(num, imax * num)
            imin = min(num, imin * num)

            # 保留整体的最大值
            max_d = max(max_d, imax)

        return max_d

    def findMin(self, nums: List[int]) -> int:
        # 每次需要比较，有点耗时
        # import sys
        # if not nums:
        #     return 0
        # lens = len(nums)
        # left, right = 0, lens
        # min_val = sys.maxsize
        # while left < right:
        #     mid = left + (right-left)//2
        #     if nums[left] < nums[mid]:
        #         min_val = min(min_val, nums[left])
        #         left = mid+1
        #     else:
        #         min_val = min(nums[mid], min_val)
        #         right = mid
        #
        # return min_val

        # 大神思路
        # 首先进行分析，发现有两种情况会收缩右边界
        # 因此只能从左边界下手
        # 右边为开区间时不方便判断右端值
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                # 此处可能会将最终值过滤掉
                right = mid

        return nums[left]

    def findPeakElement(self, nums: List[int]) -> int:
        # 实质还是暴力破解
        # lens = len(nums)
        # if lens < 2:
        #     return 0
        #
        # # 递归二分查找
        # def judge(left:int, right:int):
        #     if left < right:
        #         mid = (right-left)//2 + left
        #         if (not mid and nums[mid] > nums[mid+1]) \
        #             or (mid == lens-1 and nums[mid] > nums[mid-1]) \
        #                 or (nums[mid] > nums[mid-1] and nums[mid] > nums[mid+1]):
        #                     return mid
        #
        #         left_j = judge(left, mid)
        #         right_j = judge(mid+1, right)
        #
        #         # 找到就退出
        #         if left_j != -1:
        #             return left_j
        #         if right_j != -1:
        #             return right_j
        #     return -1
        #
        # res = judge(0, lens)
        # return res if res != -1 else 0

        # 大神思路
        # 比较m处与m+1处的值，如果m处较大，说明中点处于下坡段，则峰值在左侧
        # 反之则在右侧
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            # 看来还是有选择的空间的，只是没想到。。
            # 由于此处用到了mid+1，因此必须保持左闭右闭
            if nums[mid] > nums[mid + 1]:
                right = mid
            else:
                left = mid + 1
        return left

    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        # 每个数最多访问两遍，时间复杂度为O(n)
        lens = len(nums)
        min_len = lens
        # 左右指针标识当前满足的子序列左右界
        left, right, cur_sum = 0, 0, 0
        for i in range(lens):
            cur_sum += nums[i]
            right += 1
            # 一旦已经满足条件，则尝试缩小左指针
            while cur_sum - nums[left] >= s:
                cur_sum -= nums[left]
                left += 1

            if cur_sum >= s:
                min_len = min(min_len, right - left)

        # 注意到如果要求目标过大，则直接返回零
        return min_len if cur_sum >= s else 0

    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        res = []
        cur_path = []

        def findall(index: int, cur_n: int):
            if len(cur_path) == k and cur_n == n:
                # 浅拷贝
                res.append(cur_path[:])
                return

            for i in range(index, 10):
                # 剪去无用组合
                if cur_n != n and len(cur_path) == k:
                    return

                cur_path.append(i)
                # 直接添加由于栈的原因，在回溯时会有错误
                # 因此回溯时需要清理
                cur_n += i
                findall(i + 1, cur_n)
                # 如果直接在递归函数中修改则系统自动清理，无需回溯
                # findall(i + 1, cur_n+i)
                cur_path.pop()
                cur_n -= i

        findall(1, 0)
        return res

    def summaryRanges(self, nums: List[int]) -> List[str]:
        res = []
        if not nums:
            return res

        # 尾数添加
        nums.append(-1)

        # 使用双指针，一个负责上一个值的保存，一个负责当前区间的左边部分的保存
        pre, left = nums[0], nums[0]
        for cur_val in nums[1:]:
            # 当前差值为1时，加入当前区间
            # 为了防止溢出，需要将此处优化为 cur_val-1 == pre
            # 当然，py没有这个问题
            if cur_val - pre == 1:
                pre = cur_val
            else:
                # 否则判别当前区间是否只有一个值
                res.append(str(left))
                if left != pre:
                    res[-1] += '->' + str(pre)

                # 更新两个指针
                left = pre = cur_val

        return res

    def majorityElement(self, nums: List[int]) -> List[int]:
        # 空间复杂度为O(n)的方法：使用字典
        # lens = len(nums)
        # from collections import Counter
        # count = Counter(nums).items()
        # res = []
        #
        # for key, value in count:
        #     if value * 3 > lens:
        #         res.append(key)
        #
        # return res

        # 摩尔投票法：用于选择列表中超过半数的元素
        # 解决思路：出现相同的次数增一，不同的减一，如果减为0，则更换元素，并将次数置一

        # 此处需要对摩尔投票法进行改良，可以找出列表中超过1/（m+1）的元素，其中列表长为m
        # 抵消阶段
        # 自己写的存在问题 [1,1,1,2,3,4,5,6] 无法通过
        # if not nums:
        #     return []
        # temp = [[nums[0], 1]]
        # for cur in nums[1:]:
        #     if cur == temp[0][0]:
        #         temp[0][1] += 1
        #     else:
        #         if len(temp) == 1:
        #             temp.append([cur, 1])
        #         elif cur == temp[1][0]:
        #             temp[1][1] += 1
        #         else:
        #             for i in range(2):
        #                 if not temp[i][1]:
        #                     temp[i][0], temp[i][1] = cur, 1
        #                     break
        #                 else:
        #                     temp[i][1] -= 1
        #
        # # 还有计数阶段
        # lens = len(nums)
        # return [x[0] for x in temp if nums.count(x[0])*3 > lens]

        lens = len(nums)
        if not lens:
            return []

        # 消除阶段
        temp = [[0, 0], [-1, 0]]
        for cur in nums:
            # 当前元素在第一块
            if temp[0][0] == cur:
                temp[0][1] += 1
                continue

            # 当前元素在第二块
            if temp[1][0] == cur:
                temp[1][1] += 1
                continue

            # 第二块的次数为0时进行更换
            if not temp[1][1]:
                temp[1][0], temp[1][1] = cur, 1
                continue

            # 第一块的次数为0时进行更换
            if not temp[0][1]:
                temp[0][0], temp[0][1] = cur, 1
                continue

            # 两个块都不为0时，同时将各自次数减去1
            temp[0][1] -= 1
            temp[1][1] -= 1

        # 计数阶段
        # 最多遍历两遍
        return [x[0] for x in temp if nums.count(x[0]) * 3 > lens]

    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # 左右子数组的方法，有点类似于分而治之
        # lens = len(nums)
        # # 将最终的结果转化为先求取左右两边的结果
        # left, right = [1] + [0] * (lens - 1), [0] * (lens - 1) + [1]
        # for i in range(1, lens):
        #     left[i] = left[i - 1] * nums[i - 1]
        #
        # for i in range(lens - 2, -1, -1):
        #     right[i] = right[i + 1] * nums[i + 1]
        #
        # # 最后再相乘即可
        # return [left[t] * right[t] for t in range(lens)]

        # 将空间复杂度降为O(1)的技巧：使用结果列表
        # 有点动态规划的感觉
        lens = len(nums)
        res = [1] + [0] * (lens - 1)

        # 先算左边
        for i in range(1, lens):
            res[i] = res[i - 1] * nums[i - 1]

        # print(res)
        # 再算右边，右边需要进行统计
        # 关键点：变量统计当前的累积，没想到
        right = 1
        for j in range(lens - 1, -1, -1):
            res[j] *= right
            right *= nums[j]

        return res

    def findDuplicate(self, nums: List[int]) -> int:
        # 只能想到暴力法
        # 类比于查找环状链表的入环节点
        # 由于重复元素的存在，因此造成沿着列表中的值递归向下查找会陷入一个循环
        # 对于案例：1,3,4,2,2 可以得到路径 1-3-2-4-2-4-2...
        # 因此可以使用快慢指针来分别指向，直到两个指针指向相同值时停止
        #

        slow, quick = nums[0], nums[0]
        # 此时两者相对距离：乌龟i,兔子2i, i-ks(s为环大小)
        while True:
            slow = nums[slow]
            quick = nums[nums[quick]]

            if slow == quick:
                break

        # 此时让另一个乌龟从起点开始走，当走了i步以后，两个乌龟同时走回圈起始点
        repeat = nums[0]
        while repeat != slow:
            repeat = nums[repeat]
            slow = nums[slow]

        return slow

    def gameOfLife(self, board: List[List[int]]) -> None:
        if not board:
            return
        len_row, len_col = len(board), len(board[0])

        # 八个方向
        ways = [[0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1]]
        # change = []
        for i in range(len_row):
            for j in range(len_col):
                cur_sum = 0
                for way in ways:
                    # 累加周边细胞
                    temp_row, temp_col = i + way[0], j + way[1]
                    if 0 <= temp_row < len_row and 0 <= temp_col < len_col \
                            and abs(board[temp_row][temp_col]) == 1:
                        cur_sum += abs(board[temp_row][temp_col])

                # 需要用额外列表保存当前的变化位置（可优化）
                # if (board[i][j] and (cur_sum < 2 or cur_sum > 3)
                #         or (not board[i][j] and cur_sum == 3)):
                #      change.append([i, j])
                # way2 add status
                if board[i][j] and (cur_sum < 2 or cur_sum > 3):
                    # 过去是活的，现在挂了
                    board[i][j] = -1

                if not board[i][j] and cur_sum == 3:
                    # 过去挂了，现在活的
                    board[i][j] = 2

        # 最后进行异或即可
        # for pos in change:
        #     board[pos[0]][pos[1]] ^= 1

        # 直接对状态进行更改
        for i in range(len_row):
            for j in range(len_col):
                if board[i][j] == -1:
                    board[i][j] = 0
                if board[i][j] == 2:
                    board[i][j] = 1

        print(board)

    def findDuplicates(self, nums: List[int]) -> List[int]:
        # 字典法
        # from collections import Counter
        # res = Counter(nums).items()
        #
        # return [key for key,value in res if value == 2]

        # 下标余数法
        # 对于每一个值指向的下标位置增加整体长度n，然后找出超过2n的即可
        # 想不出来（死记死记）
        lens = len(nums)
        for i in range(lens):
            nums[(nums[i] - 1) % lens] += lens

        return [index + 1 for index, value in enumerate(nums) if value > 2 * lens]

    def circularArrayLoop(self, nums: List[int]) -> bool:
        # 想到快慢指针，但是对其的用法知之甚少
        # 遇到环的问题优先考虑快慢指针
        lens = len(nums)
        if not lens:
            return False

        # 将不在环中的点置零
        def set_zero(i: int):
            while True:
                j = (i + nums[i] + 5000 * lens) % lens
                # 若下一位为0或者两者异号则停止
                if nums[j] == 0 or nums[j] ^ nums[i]:
                    nums[i] = 0
                    break
                # 否则一直向下递推
                nums[i] = 0
                i = j

        for k in range(lens):
            # 剪枝
            if nums[k] == 0:
                continue

            # 快指针走两步，慢指针走一步，如果有环，则一定会重合，否则全体都被置零
            slow, quick = k, k
            while True:
                pre_slow = slow
                slow = (slow + nums[slow] + 5000 * lens) % lens
                # 出现当前点与之前点重合（表示环长度为1），两点异号或者当前点为0时，
                # 说明这一系列都不在环中，需要递归置零
                if pre_slow == slow or nums[slow] ^ nums[pre_slow] < 0 or nums[slow] == 0:
                    set_zero(k)
                    break

                pre_quick = quick
                quick = (quick + nums[quick] + 5000 * lens) % lens
                if pre_quick == quick or nums[quick] ^ nums[pre_quick] < 0 or nums[quick] == 0:
                    set_zero(k)
                    break

                pre_quick = quick
                quick = (quick + nums[quick] + 5000 * lens) % lens
                if pre_quick == quick or nums[quick] ^ nums[pre_quick] < 0 or nums[quick] == 0:
                    set_zero(k)
                    break

                # 两指针重合
                if quick == slow:
                    return True

        return False

    def findPoisonedDuration(self, timeSeries: List[int], duration: int) -> int:
        # 贪心思想
        sum_time = 0
        lens = len(timeSeries)
        # 边界判断
        if not lens:
            return 0
        i = 0
        while i < lens - 1:
            # 可优化为一句
            sum_time += min(duration, timeSeries[i + 1] - timeSeries[i])
            # if cur_pos < timeSeries[i+1]:
            #     sum_time += duration
            # else:
            #     sum_time += timeSeries[i+1]-timeSeries[i]

            i += 1

        return sum_time + duration

    def subarraySum(self, nums: List[int], k: int) -> int:
        # 方法一：暴力法
        # 显然超时
        # count = 0
        # lens = len(nums)
        # for i in range(lens):
        #     for j in range(i, lens):
        #         if sum(nums[i:j+1]) == k:
        #             count += 1
        #
        # return count

        # 方法二：使用额外空间存储中间计算（使用累计和列表保存计算）
        # 还是超时，还需要进一步优化
        # lens = len(nums)
        # count = 0
        # sub_arr = [nums[0]] + [0]*(lens-2)
        # # 此处会漏掉第一个解
        # if sub_arr[0] == k:
        #     count += 1
        # # 由于此循环仅在sub_arr长度超过1时才进入
        # for i in range(lens-1):
        #     if not i:
        #         sub_arr[i] = sub_arr[i]+nums[i+1]
        #     else:
        #         sub_arr[i] = sub_arr[i-1]+nums[i+1]
        #     # 第一步
        #     if sub_arr[i] == k:
        #         count += 1
        # # 此处还是很耗时
        # for i in range(lens):
        #     for j in range(i, lens-1):
        #         sub_arr[j] -= nums[i]
        #         if sub_arr[j] == k:
        #             count += 1
        #
        # return count

        # 方法三 哈希法(完全想不到) ***
        # 前缀和思想
        # 关键点：利用等式sum[r]-sum[l]=k进行变形
        # 利用sum[r]-sum[l]即为l到r区间的和值来处理
        # 若出现 sum[r]-sum[l]=k，说明含有和为k的子数组
        count, cur_sum, lens = 0, 0, len(nums)
        # 初始值有可能为k
        # 字典中的key存储前缀和，value存储的是前缀和出现的次数
        dicts = {0: 1}

        for i in range(lens):
            # 统计当前的和
            cur_sum += nums[i]
            # 实质为 sum[i]-k == sum[pre]
            if cur_sum - k in dicts:
                count += dicts[cur_sum - k]

            # 累计前缀和次数
            if cur_sum in dicts:
                dicts[cur_sum] += 1
            else:
                dicts[cur_sum] = 1

        return count

    def arrayNesting(self, nums: List[int]) -> int:
        # 标记法：时间复杂度为O(n)，每个元素最多访问两次
        # 空间复杂度为O(n)，需要一个访问列表
        # lens = len(nums)
        # if not lens:
        #     return 0
        #
        # visit = [0] * lens
        # count_max, cur_count = 0, 0
        # for i in range(lens):
        #     # 向下递归遍历
        #     while not visit[i]:
        #         cur_count += 1
        #         visit[i] = 1
        #         i = nums[i]
        #
        #     # 选取其中的最大值即可
        #     count_max = max(count_max, cur_count)
        #     # 剪枝效果不是很理想
        #     # if cur_count >= lens//2+1:
        #     #     break
        #     cur_count = 0
        #
        # return count_max

        # 方法二：直接标记：无需标记列表
        lens = len(nums)
        if not lens:
            return 0

        count_max, cur_max = 0, 0
        for i in range(lens):
            while nums[i] > -1:
                cur_max += 1
                temp = i
                i = nums[i]
                nums[temp] = -1

            count_max = max(count_max, cur_max)
            cur_max = 0

        return count_max

    def leastInterval(self, tasks: List[str], n: int) -> int:
        # 使用列表来存储当前出现的任务次数
        arr = [0] * 26
        for ch in tasks:
            arr[ord(ch) - ord('A')] += 1

        arr.sort(reverse=True)
        # 方法一：公式法
        # max_work, same = arr[0], 0
        # # 统计出现最多次数任务的数量
        # for cur in arr:
        #     if max_work == cur:
        #         same += 1
        #     else:
        #         break
        #
        # # 最终结果等于
        # # 1 最多数量任务-1的n+1倍（自身也需要计算时间）
        # # 2 最多数量任务的数量（可能有任务A,B都出现4次），此项直接置于后方
        # # 3 剩余工作总量-剩余位置，出现负数时表示有待命状态
        # return (n+1)*(max_work-1)+same+max(0, len(tasks)-max_work-n*(max_work-1)-same+1)

        # 方法二：演示法
        count = 0
        while arr[0] > 0:
            for i in range(n + 1):
                # 执行完成则结束
                if not arr[0]:
                    break
                # 如果一次可以执行的任务数量较多，则用于待命状态
                if i < 26 and arr[i] > 0:
                    arr[i] -= 1
                count += 1

            # 保证每次都从当前剩余最多任务数量处开始处理
            arr.sort(reverse=True)

        return count

    def triangleNumber(self, nums: List[int]) -> int:
        # 回溯法：不剪枝会超时，能想到的剪枝中也会超时
        # lens = len(nums)
        # if lens < 3:
        #     return 0
        #
        # nums.sort()
        # elems = []
        #
        # def find_tuple(index:int):
        #     if len(elems) == 3:
        #         if elems[0]+elems[1] > elems[2]:
        #             self.count += 1
        #         return
        #
        #     for i in range(index, lens):
        #         if len(elems) == 2 and elems[0]+elems[1] <= nums[i]:
        #             return
        #
        #         elems.append(nums[i])
        #         find_tuple(i+1)
        #         elems.pop()
        #
        # find_tuple(0)
        # return self.count

        # 方法二：循环解决:还是没想到点子上O(n3)
        # lens = len(nums)
        # if lens < 3:
        #     return 0
        # count = 0
        # nums.sort()
        # for i in range(lens-2):
        #     if nums[i] == 0:
        #         continue
        #     for j in range(i+1, lens-1):
        #         if nums[j] == 0:
        #             continue
        #
        #         # 将k从右向左取
        #         k = lens-1
        #         while nums[k] >= nums[i]+nums[j]:
        #             k -= 1
        #         count += k-j
        #         # k = j+1
        #         # while k < lens and nums[i]+nums[j] > nums[k]:
        #         #     count += 1
        #         #     k += 1
        #
        # return count

        # 方法三：双指针法
        nums.sort()
        lens = len(nums)
        count = 0

        # 充分利用从大取到小的思想
        for i in range(lens - 1, 1, -1):
            l, r = 0, i - 1
            while l < r:
                # 两小端最小值处都满足条件，说明r-l处的皆满足条件
                if nums[l] + nums[r] > nums[i]:
                    count += r - l
                    r -= 1
                # 说明两小端过小，需要增大最小端
                else:
                    l += 1

        return count

    def maximumSwap(self, num: int) -> int:
        # 无法通过的案例：99901
        # res = []
        # while num:
        #     res.append(num % 10)
        #     num //= 10
        #
        # lens = len(res)
        # exc = [0, 0]
        #
        # 先从后向前找出当前最大的值
        # for i in range(lens-1, 0, -1):
        #     if res[i-1] >= res[i] and res[i-1]>=exc[0]:
        #         exc = [res[i-1], i-1]
        # 再从右向前找出第一个比最大值小的数，并且该数下标要更大
        # for i in range(lens-1, -1, -1):
        #     if res[i] < exc[0] and i > exc[1]:
        #         res[i], res[exc[1]] = res[exc[1]], res[i]
        #         break
        #
        # result = 0
        # for i in range(lens):
        #     result += (10**i)*res[i]
        #
        # return result

        # 自己的算法根本考虑不到这么多情况
        # 排序法简单明了

        res = []
        while num:
            res.append(num % 10)
            num //= 10

        lens = len(res)
        res_sort = sorted(res)
        swap_pos, value = lens, -1
        # 与排序序列比较，第一个不相同的位置必为交换位，并且有序序列的该位置为交换的值
        for i in range(lens):
            if res[i] != res_sort[i]:
                swap_pos, value = i, res_sort[i]

        # 从左向右查找值的位置，然后进行交换即可
        for i in range(lens):
            if value == res[i]:
                res[i], res[swap_pos] = res[swap_pos], res[i]

        # 累加
        result = 0
        for i in range(lens):
            result += (10 ** i) * res[i]

        return result

    def constructArray(self, n: int, k: int) -> List[int]:
        # 使用一个set来判断该值是否使用
        # 时间复杂度为O(n)，空间复杂度为O(n)
        arr_set = set(i for i in range(2, n + 1))
        cur_pos = 0
        res = [1]
        while k != 1:
            # 绝对值添加最好是加减交替进行
            # 防止出现跨度过大的值
            temp = res[cur_pos] + k
            if cur_pos % 2:
                temp = res[cur_pos] - k

            res.append(temp)
            cur_pos += 1
            k -= 1
            arr_set.remove(temp)

        # 顺序添加剩余值即可
        while arr_set:
            res.append(arr_set.pop())

        return res

    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        # 又一次被变量的作用域给坑死，注意吸取教训啊
        # 将全局变量替换为局部变量更好（使用递归的返回）
        # 说明在细节方面还有待提高
        # 常规解法，遇到能遍历的直接bfs即可
        if not grid:
            return 0

        len_row, len_col = len(grid), len(grid[0])
        visit = [[0] * len_col for _ in range(len_row)]
        directions = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        # 由于不需要在内部函数中使用，因此可以不使用 __init__函数
        max_area = 0

        def dfs(pos_x: int, pos_y: int):
            for k in range(4):
                temp_x, temp_y = pos_x + directions[k][0], pos_y + directions[k][1]
                if 0 <= temp_x < len_row and 0 <= temp_y < len_col \
                        and not visit[temp_x][temp_y] and grid[temp_x][temp_y]:
                    visit[temp_x][temp_y] = 1
                    self.cur_area += 1
                    dfs(temp_x, temp_y)
                else:
                    # 此处应该能优化
                    self.max_area = max(self.max_area, self.cur_area)

        def dfs_2(pos_x: int, pos_y: int) -> int:
            if not (0 <= pos_x < len_row and 0 <= pos_y < len_col) or visit[pos_x][pos_y] \
                    or not grid[pos_x][pos_y]:
                return 0

            # 直接对原始矩阵置零，省去了访问列表的构建（妙啊）
            grid[pos_x][pos_y] = 0
            cur_area = 1
            for i in range(4):
                temp_x, temp_y = pos_x + directions[i][0], pos_y + directions[i][1]
                cur_area += dfs_2(temp_x, temp_y)

            return cur_area

        # 方法一：使用构造器
        # for i in range(len_row):
        #     for j in range(len_col):
        #         if grid[i][j] and not visit[i][j]:
        #             self.cur_area = 1
        #             visit[i][j] = 1
        #             dfs(i, j)

        # 方法二：局部变量的使用（更好）
        for i in range(len_row):
            for j in range(len_col):
                if grid[i][j]:
                    # 逐个求取即可
                    max_area = max(max_area, dfs_2(i, j))

        return max_area

    def maxProfit(self, prices: List[int], fee: int) -> int:
        # 动态规划思想
        # 使用变量描述状态（状态机）：是个很好的技巧，还是得多用
        # 第一个状态为天数，第二个状态为是否持有股票
        # lens = len(prices)
        # dp = [[0, 0] for _ in range(lens)]
        # dp[0][1] = -prices[0]
        #
        # for i in range(1, lens):
        #     dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i] - fee)
        #     dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
        #
        # return dp[-1][0]

        # 在进行基本实现后可以考虑空间优化
        lens = len(prices)
        own, not_own = -prices[0], 0
        for i in range(1, lens):
            # 注意变量的替换即可
            temp = own
            own = max(own, not_own - prices[i])
            not_own = max(not_own, temp + prices[i] - fee)

        return not_own

    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        # 最开始的时候想过前缀和，但是感觉会溢出，因此考虑双指针
        # 尝试双指针, perfect
        # 左右指针之间的为当前累积小于k的最长子列表
        # 有点类似于动态规划的思想，每次加入一个元素再进行调整
        left, right = 0, 0
        count, lens, cur_mul = 0, len(nums), 1
        while right < lens:
            if nums[right] < k:
                cur_mul *= nums[right]
                # 如果累乘后结果过大，则需要移动左指针
                while cur_mul >= k:
                    cur_mul //= nums[left]
                    left += 1
                # 可以推导，每次加入一个元素时所能产生满足条件的子列表数为下式，包含自身
                count += right - left + 1
            else:
                # 如果当前单值都比k大，则整体右移
                cur_mul = 1
                left = right + 1
            # 在不超过k时，仅需要移动右指针
            right += 1

        return count

    def findLength(self, A: List[int], B: List[int]) -> int:
        # 思路一：用字典存储下标再进行查找
        # 跟暴力差不多了，超时
        # dicts = {}
        # for index, e in enumerate(A):
        #     if e in dicts:
        #         dicts[e].append(index)
        #     else:
        #         dicts[e] = [index]
        #
        # len_A, len_B = len(A), len(B)
        # max_len = 0
        # for index, e in enumerate(B):
        #     if e in dicts:
        #         for pos in dicts[e]:
        #             i = 0
        #             while pos + i < len_A and index + i < len_B and A[pos + i] == B[index + i]:
        #                 i += 1
        #             max_len = max(max_len, i)
        #
        # return max_len

        # 经典的动态规划问题：？？？（以前都没听说过）
        # 状态量为子数组的长度
        len_A, len_B = len(A), len(B)
        # 需要增加一行一列作为起始，因为转移方程中初始需要-1
        dp = [[0]*(len_B+1) for _ in range(len_A+1)]

        max_len = 0
        for i in range(1, len_A+1):
            for j in range(1, len_B+1):
                # 对应方式为下标+1
                if A[i-1] == B[j-1]:
                    # 状态转移方程：当前的最长长度为前一段的长度加上当前长度
                    dp[i][j] = dp[i-1][j-1]+1

                max_len = max(max_len, dp[i][j])
        return max_len


# 二分查找最优方法，保留左闭右开原则
def binary_sarch(arr: List[int], target: int) -> int:
    left, right = 0, len(arr)

    # 无需判断相等的情况，只需要在最后判别结果是否真正存在即可
    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid

    return left if left < len(arr) and arr[left] == target else -1


if __name__ == '__main__':
    show = Solution()

    # 11 盛最多水的容器
    # print(show.maxArea([1, 8, 6, 2, 5, 4, 8, 3, 7]))

    # 15 三数之和
    # print(show.threeSum([-4, -2, -2, -2, 0, 1, 2, 2, 2, 3, 3, 4, 4, 6, 6]))

    # 16 最接近的三数之和
    # print(show.threeSumClosest([1,2,4,8,16,32,64,128], 82))

    # 18 四数之和
    # print(show.fourSum([5, 5, 3, 5, 1, -5, 1, -2], 4))

    # 31 下一个排列
    # print(show.nextPermutation([4,2,0,2,3,2,0]))

    # 33 搜索旋转排序数组
    # print(show.search([6,5,4,3,2,1,0], 2))

    # 34 在排序数组中查找元素的第一个和最后一个位置
    # print(show.searchRange([2,2], 6))

    # 39 组合总和
    # print(show.combinationSum([2,3,7], 18))

    # 40 组合总和II
    # print(show.combinationSum2([4,2,5,2,5,3,1,5,2,2], 9))

    # 48 旋转图像
    # print(show.rotate([[5, 1, 9, 11],
    #                    [2, 4, 8, 10],
    #                    [13, 3, 6, 7],
    #                    [15, 14, 12, 16]]))

    # 54 螺旋矩阵
    # print(show.spiralOrder([
    #     [1, 2, 3, 4],
    #     [5, 6, 7, 8],
    #     [9, 10, 11, 12]
    # ]))

    # 55 跳跃游戏
    # print(show.canJump([3,2,1,0,4]))

    # 56 合并区间
    # print(show.merge([[1,5],[2,3]]))

    # 59 螺旋矩阵II
    # print(show.generateMatrix(10))

    # 62 不同路径
    # print(show.uniquePaths(7, 3))

    # 63 不同路径II
    # print(show.uniquePathsWithObstacles([[0]]))

    # 64 最小路径和
    # print(show.minPathSum([[1,3,1],[1,5,1],[4,2,1]]))

    # 73 矩阵置零
    # print(show.setZeroes([
    #     [0, 1, 2, 0],
    #     [3, 4, 5, 2],
    #     [1, 3, 1, 5]]))

    # 74 搜索二维矩阵
    # print(show.searchMatrix([
    #     [1, 3, 5, 7],
    #     [10, 11, 16, 20],
    #     [23, 30, 34, 50]
    # ], 3))

    # 75 颜色分类
    # print(show.sortColors([2,0,1]))

    # 78 子集
    # print(show.subsets([1,2,3]))

    # 79 单词搜索
    # print(show.exist([
    #     ['A', 'B', 'C', 'E'],
    #     ['S', 'F', 'C', 'S'],
    #     ['A', 'D', 'E', 'E']
    # ], 'SECCS'))

    # 80 删除排序数组中的重复项II
    # print(show.removeDuplicates([0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4]))

    # 81 搜索旋转排序数组II
    # print(show.search([1,2,1,1], 2))

    # 90 子集II
    # print(show.subsetsWithDup([1,2,2]))

    # 105 从前序与中序遍历序列构造二叉树
    # show.travel_tree_bfs(show.buildTree([3,1,2,4], [1,2,3,4]))

    # 106 从中序与后序遍历序列构造二叉树
    # show.travel_tree_bfs(show.buildTree([1,2], [2,1]))

    # 120 三角形最小路径和
    # print(show.minimumTotal([
    #     [2],
    #     [3, 4],
    #     [6, 5, 7],
    #     [4, 1, 8, 3]
    # ]))

    # 152 乘积最大子数组
    # print(show.maxProduct([-2,1,0, -3,1,2,-1]))

    # 153 寻找旋转排序数组中的最小值
    # print(show.findMin([5,1,2,3,4]))

    # 162 寻找峰值
    # print(show.findPeakElement([1,2,1,3,5,6,4]))

    # 209 长度最短的子数组
    # print(show.minSubArrayLen(5, [2, 3, 1, 2, 4, 3]))

    # 216 组合总和III
    # print(show.combinationSum3(3, 9))

    # 228 汇总区间
    # print(show.summaryRanges([0, 2, 3, 4, 6, 8, 9]))

    # 229 求众数II
    # print(show.majorityElement([1,1,1,2,3,4,5,6]))

    # 238 除自身外数组的乘积
    # print(show.productExceptSelf([1, 2, 3, 4]))

    # 287 寻找重复数***（弗洛伊德的乌龟和兔子，也称为循环检测）
    # print(show.findDuplicate([3, 1, 3, 4, 2]))

    # 289 生命游戏
    # print(show.gameOfLife([
    #     [0, 1, 0],
    #     [0, 0, 1],
    #     [1, 1, 1],
    #     [0, 0, 0]
    # ]))

    # 380 常数时间插入删除和获取随机元素
    # randomset = RandomizedSet()
    # print(randomset.insert(0))
    # print(randomset.insert(1))
    # print(randomset.remove(0))
    # print(randomset.insert(2))
    # print(randomset.remove(1))
    # print(randomset.getRandom())

    # 442 数组中的重复数据
    # print(show.findDuplicates([4, 3, 2, 7, 8, 2, 3, 1]))

    # 457 环形数组循环
    # print(show.circularArrayLoop([3,1,2]))

    # 495 提莫攻击
    # print(show.findPoisonedDuration([1, 2, 4, 8], 3))

    # 560 和为k的子数组
    # print(show.subarraySum([1,1],1))

    # 565 数组嵌套
    # print(show.arrayNesting([5, 4, 0, 3, 1, 6, 2]))

    # 621 任务调度器
    # print(show.leastInterval(["A","A","A","B","B","V","V","C","C"], 2))

    # 611 有效三角形个数
    # print(show.triangleNumber([0, 1, 0, 1, 2, 2, 3, 4, 5]))

    # 670 最大交换
    # print(show.maximumSwap(99901))

    # 667 优美的排列
    # print(show.constructArray(3, 1))

    # 695 岛屿的最大面积
    # print(show.maxAreaOfIsland([[1, 1, 1], [1, 0, 0]]))

    # 714 买卖股票的最佳时机含手续费
    # print(show.maxProfit([1, 5, 2, 9, 2, 6, 5, 7], 2))

    # 713 乘积小于k的子数组
    # print(show.numSubarrayProductLessThanK([10, 5, 2, 12, 1], 11))

    # 718 最长重复子数组
    # print(show.findLength([1,2,4,5], [2,3,1,2,4,5]))

    # 729 我的日程安排表I
    # test = MyCalendar()
    # print(test.book(10, 20))
    # print(test.book(15, 25))
    # print(test.book(20, 30))
    # print(test.book(2, 10))
    # print(test.book(1, 3))

