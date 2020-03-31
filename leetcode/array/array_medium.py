from typing import List


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def __init__(self):
        self.order = 0

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
            triangle[i][0] += triangle[i-1][0]
            triangle[i][-1] += triangle[i-1][-1]

            # 动态规划思想
            for j in range(1, len_cur-1):
                triangle[i][j] += min(triangle[i-1][j], triangle[i-1][j-1])

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
            imax = max(num, imax*num)
            imin = min(num, imin*num)

            # 保留整体的最大值
            max_d = max(max_d, imax)

        return max_d


# 二分查找最优方法，保留左闭右开原则
def binary_search(arr: List[int], target: int) -> int:
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
