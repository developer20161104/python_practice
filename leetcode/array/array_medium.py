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
            mid = left + (right - left) // 2
            if end > self.time_arr[mid][0]:
                left = mid + 1
            else:
                right = mid

        # 判断标准为上一个日程的end与当前待插入的st
        # 如果st较小说明此日程与上一个日程重叠
        if left > 0 and self.time_arr[left - 1][1] > start:
            return False
        else:
            # 否则直接插入位置即可
            self.time_arr.insert(left, [start, end])
            return True


# 900 RLE迭代器
class RLEIterator:
    # 直接按照题目要求进行保存内存会爆炸，因此需要其他的办法
    # wrong：内存溢出
    # def __init__(self, A: List[int]):
    #     self.arr = []
    #     self.pos = -1
    #
    #     for i in range(0, len(A) - 1, 2):
    #         self.arr += [A[i + 1]] * A[i]
    #     self.lens = len(self.arr)
    #
    # def next(self, n: int) -> int:
    #     self.pos += n
    #     if self.pos >= self.lens:
    #         return -1
    #     return self.arr[self.pos]

    # 主要思想：使用下标来标记当前使用到的位置，
    # 并在原始列表中修改使用后的剩余值，减少额外开销
    def __init__(self, A: List[int]):
        self.A = A
        self.pos = 0

    def next(self, n: int) -> int:
        if not self.A or self.pos >= len(self.A):
            return -1
        temp = self.A[self.pos] - n

        if temp >= 0:
            self.A[self.pos] = temp
            return self.A[self.pos + 1]
        else:
            # 此处的循环真滴难整
            self.pos += 2
            # 两者的结束条件位置是不一样的
            while self.pos < len(self.A):
                temp += self.A[self.pos]
                if temp >= 0:
                    break
                self.pos += 2

            if self.pos < len(self.A):
                self.A[self.pos] = temp
                return self.A[self.pos + 1]
            else:
                return -1


# 1146 快照数组
class SnapshotArray:
    # 查询复杂度都为O(1)还是超时可还行
    # 但是空间复杂度过于夸张，需要优化，不能每次都将整体进行保存

    # # 有一点隐藏错误，题目没有给出，就是当下一次保存之前列表没有做任何修改，则保存的是上一次的列表，而不是更新列表
    # # 初始化一个与指定长度相等的类数组的数据结构
    # def __init__(self, length: int):
    #     self.len = length
    #     self.curlist = [0]*self.len
    #     self.count = 1
    #     self.search_dict = {}
    #
    # # 指定索引处的元素设置
    # def set(self, index: int, val: int) -> None:
    #     self.curlist[index] = val
    #
    # # 获取数组的快照，并返回快照编号
    # def snap(self) -> int:
    #     self.search_dict[self.count-1] = self.curlist[:]
    #     self.count += 1
    #     # self.curlist = [0]*self.len
    #
    #     return self.count-2
    #
    # # 根据指定id选择快照，并返回快照的索引值
    # def get(self, index: int, snap_id: int) -> int:
    #     if snap_id in self.search_dict:
    #         return self.search_dict[snap_id][index]
    #     return -1

    # 另一种思路：每次快照只保存修改的位置，并且以二维列表的形式保存在对应位置上
    # 在进行查找时使用二分降低时间复杂度
    def __init__(self, length: int):
        # 此处有问题，必须考虑不插入即保存的情况
        self.arr = [{0: 0} for _ in range(length)]
        self.count = 0

    # 指定索引处的元素设置
    def set(self, index: int, val: int) -> None:
        self.arr[index][self.count] = val

    # 获取数组的快照，并返回快照编号
    def snap(self) -> int:
        self.count += 1
        return self.count - 1

    # 根据指定id选择快照，并返回快照的索引值
    def get(self, index: int, snap_id: int) -> int:
        # 在进行查找时，由于保存的是每次的修改值，因此有可能出现所查值不在保存列表中
        # 因此还需要进行二分查找，寻找最近的修改
        if snap_id in self.arr[index]:
            return self.arr[index][snap_id]

        search_list = list(self.arr[index].keys())
        left, right = 0, len(search_list)
        while left < right:
            mid = left + (right - left) // 2
            if search_list[mid] < snap_id:
                left = mid + 1
            else:
                right = mid

        return self.arr[index][search_list[left - 1]]


# 1352 最后k个数的乘积
class ProductOfNumbers:
    # 使用前缀积会有问题，如果出现0时，后续全部为0
    # 因此，当num为0时，需要对保存的值修改为1，并记录当前0出现的位置
    # 可以通过预设长度来进一步优化时间复杂度
    def __init__(self):
        self.pre_mul = [1]
        self.len = 1
        self.zero_pos = -1

    def add(self, num: int) -> None:
        # 就是此处有点耗时，每次添加需要使用O(n)的时间来创建
        if not num:
            self.zero_pos = self.len - 1
            self.pre_mul.append(1)
        else:
            self.pre_mul.append(self.pre_mul[self.len - 1] * num)
        self.len += 1

    def getProduct(self, k: int) -> int:
        # 如果长度范围内出现0，则直接返回0，否则，利用前缀积可以在O(1)的时间内求出结果
        return 0 if self.len - k - 1 <= self.zero_pos else self.pre_mul[self.len - 1] // self.pre_mul[self.len - 1 - k]


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
        dp = [[0] * (len_B + 1) for _ in range(len_A + 1)]

        max_len = 0
        for i in range(1, len_A + 1):
            for j in range(1, len_B + 1):
                # 对应方式为下标+1
                if A[i - 1] == B[j - 1]:
                    # 状态转移方程：当前的最长长度为前一段的长度加上当前长度
                    dp[i][j] = dp[i - 1][j - 1] + 1

                max_len = max(max_len, dp[i][j])
        return max_len

    def maxChunksToSorted(self, arr: List[int]) -> int:
        # 求环的数量？ wrong : 3 2 4 0 1 5
        # count = 0
        # if not arr:
        #     return count
        #
        # lens = len(arr)
        # cur_pos = arr.index(0)
        # if cur_pos == lens-1:
        #     return 1
        #
        # while cur_pos < lens:
        #     while arr[cur_pos] != -1:
        #         temp = cur_pos
        #         arr[cur_pos], cur_pos = -1, arr[temp]
        #     count += 1
        #     cur_pos += 1
        #
        # return count

        # 采用字典来存储每个列表元素对应的下标
        # 贪心思想，时间复杂度为O(n)
        # dicts = {}
        # for index, e in enumerate(arr):
        #     dicts[e] = index
        #
        # # cur_max 保存当前子块的最短长度
        # # i 为当前元素值，一直遍历到cur_max停止
        # count, i, cur_max = 0, 1, dicts[0]
        # while cur_max < len(arr):
        #     while i <= cur_max:
        #         # 如果当前块中出现的元素下标超出当前最短长度，则向后延伸
        #         if dicts[i] > cur_max:
        #             cur_max = dicts[i]
        #         i += 1
        #     count += 1
        #     # 完成一个子块的构造
        #     cur_max += 1
        #
        # return count

        # 思路二：判断当前最大元素值与下标是否相等即可（这个没有想到）
        count = cur_max = 0
        for index, e in enumerate(arr):
            cur_max = max(cur_max, e)
            # 若当前最大值已经与下标相等时，说明子块已经构成（相当于我自己思路的逆向）
            if cur_max == index:
                count += 1

        return count

    def isIdealPermutation(self, A: List[int]) -> bool:
        # 全局的计算方式有问题： 2,1,0
        # wrong：在全局倒置计算上有问题
        # local_count, global_count = 0, 0
        # if len(A) < 2:
        #     return True
        # for index, e in enumerate(A):
        #     if index > 0 and A[index] - A[index - 1] < 0:
        #         local_count += 1
        #     if A[index] - index > 0:
        #         global_count += A[index] - index
        #
        # return global_count == local_count

        # 如果第i个数距离本身应在位置（顺序排列）大于1时，全局倒置一定大于局部倒置
        # 根本不知道这个玩意
        for index, e in enumerate(A):
            if abs(e - index) > 1:
                return False

        return True

    def numMatchingSubseq(self, S: str, words: List[str]) -> int:
        # 逐个搜索不知道什么原因有问题
        # from collections import Counter
        # dicts = Counter(S)
        # count = 0
        # for word in words:
        #     compare = dicts.copy()
        #     flag = True
        #     for ch in word:
        #         if ch in compare and compare[ch] > 0:
        #             compare[ch] -= 1
        #         else:
        #             flag = False
        #             break
        #
        #     if flag:
        #         count += 1
        # return count

        # 桶排序：学到老活到老
        # 将每个word置于首字符所在桶中，然后每次删减第一个字符，并将删除后的字符重新分配
        # 注意，需要在删除前将先前的桶进行清空，防止因为相同字符产生重复计数
        count = 0
        arr = [[] for _ in range(26)]
        # 桶数量设置为26，分别对应26个字母
        for word in words:
            # 先将各word放入对应的桶中
            arr[ord(word[0]) - ord('a')].append(word)

        for ch in S:
            # 须先将原始桶位置清空，防止出现相同字符时重复判断
            # 预防诸如 dd 此类情况，这一点有点没想到
            temp = arr[ord(ch) - ord('a')]
            arr[ord(ch) - ord('a')] = []
            for value in temp:
                if len(value) == 1:
                    count += 1
                else:
                    arr[ord(value[1]) - ord('a')].append(value[1:])

        return count

    def numSubarrayBoundedMax(self, A: List[int], L: int, R: int) -> int:
        # 整体思路一还是有问题，无法考虑完全
        # count = 0
        # A.append(R+1)
        # left, right, lens = 0, 0, len(A)
        # while right < lens:
        #     if L <= A[right] <= R:
        #         count += 1
        #     elif A[right] > R:
        #         # 从右向左判别
        #         temp_right = right
        #         while left <= temp_right and not L <= A[temp_right] <= R:
        #             temp_right -= 1
        #         while temp_right >= left:
        #             count += right-temp_right-1
        #             temp_right -= 1
        #         left = right+1
        #     right += 1
        #
        # return count

        # 网上思路，逆向思维法
        # 时间复杂度为O(n)，空间复杂度为O(1)
        # 既然题目要求只需要包含范围内的数字，而不能存在大于范围的数字，则可以考虑
        # 即ans = 包含元素 <= R 的区间 - 包含元素 <L 的区间
        lens = len(A)
        lower_R, lower_L = 0, 0
        left, right = 0, 0
        # 统计最大元素<=R的子数组个数
        while right < lens:
            if A[right] <= R:
                right += 1
                lower_R += right - left
            else:
                right += 1
                left = right

        left = right = 0
        # 统计最大元素<L的子数组个数
        while right < lens:
            if A[right] < L:
                right += 1
                lower_L += right - left
            else:
                right += 1
                left = right

        return lower_R - lower_L

    def numFriendRequests(self, ages: List[int]) -> int:
        arr = [0] * 121
        count = 0
        # 范围约束
        for age in ages:
            arr[age] += 1

        for age_a in range(1, 121):
            for age_b in range(1, 121):
                # 第三个条件貌似没什么用
                if age_b <= 0.5 * age_a + 7 or age_b > age_a:
                    # or age_b > 100 and age_a < 100:
                    continue

                count += arr[age_a] * arr[age_b]
                # 不能给自己发请求
                if age_a == age_b:
                    count -= arr[age_a]

        return count

    def largestOverlap(self, A: List[List[int]], B: List[List[int]]) -> int:
        # 这个题目完全不会***
        # 将两个图像的重叠部分找出来，可以转化为对偏移量的枚举，然后再进行计数
        # 但是实质还是暴力计算
        # 枚举出现1的位置，并记录行列坐标
        A_2 = [complex(r, c) for r, row in enumerate(A) for c, v in enumerate(row) if v]
        B_2 = [complex(r, c) for r, row in enumerate(B) for c, v in enumerate(row) if v]

        B_set = set(B_2)
        seen = set()
        res = 0
        for a in A_2:
            for b in B_2:
                # 首先获取当前点的偏移量
                offset = b - a
                # 对于已经存在的偏移量，内部遍历会进行过滤
                if offset not in seen:
                    seen.add(offset)
                    # 遍历A中的坐标与偏移量之和出现在B中的数量，当然也包括自身
                    # 这句话很 pythonic
                    res = max(res, sum(x + offset in B_set for x in A_2))

        return res

    def advantageCount(self, A: List[int], B: List[int]) -> List[int]:
        # 排序加二分查找，时间复杂度为O(nlogn)，空间复杂度为O(n)
        # 包含了贪心的思想
        # 也可以采用双排，但是定位原位置需要消耗额外空间，思路应该差不多
        A.sort()
        res = []
        for value in B:
            cur_len = len(A)
            left, right = 0, cur_len
            # 二分查找右边界
            while left < right:
                mid = left + (right - left) // 2
                if A[mid] <= value:
                    left = mid + 1
                else:
                    right = mid

            # # 移动到较大元素位
            # while left < cur_len and A[left] <= value:
            #     left += 1

            # 最多只需要移动一位即可
            if left < cur_len and A[left] <= value:
                left += 1

            res.append(A[left % cur_len])
            A.pop(left % cur_len)

        return res

    def lenLongestFibSubseq(self, A: List[int]) -> int:
        # 暴力法：能通过就很神奇
        # 用set保存待查询数字
        # find_sort = set(A)
        # max_len = 0
        # for st, cur_e in enumerate(A):
        #     for next_e in A[st+1:]:
        #         cur_len, cur_sum = 2, cur_e+next_e
        #         temp = next_e
        #         # 每次进行迭代查找下一位
        #         while cur_sum in find_sort:
        #             temp1 = cur_sum
        #             cur_sum += temp
        #             temp = temp1
        #             cur_len += 1
        #
        #         if cur_len < 3:
        #             cur_len = 0
        #         max_len = max(max_len, cur_len)
        #
        # return max_len

        # 动态规划思想完全想不出来（实质也是暴力），只不过是保存了中间状态，比较聪明的暴力
        # 定义 状态dp[i][j]表示以A[i]A[j]结尾的斐波拉契数列
        # 得到方程 dp[i][j] = max(dp[k][i]+1) , A[k]+A[i]=A[j]
        # 并且需要建立值到索引的映射
        lens = len(A)
        max_len = 0

        # 建立值到下标的映射，减少查找时间
        dicts = {e: index for index, e in enumerate(A)}
        dp = [[0] * lens for _ in range(lens)]

        # 初始化长度
        for i in range(lens):
            for j in range(i + 1, lens):
                dp[i][j] = 2

        for i in range(lens):
            for j in range(i + 1, lens):
                # 查找A[k]是否存在，并且k应该要小于i才合理
                # 有了状态转移方程，代码就很好解决了
                ak = A[j] - A[i]
                if ak in dicts and dicts[ak] < i:
                    dp[i][j] = dp[dicts[ak]][i] + 1

                max_len = max(max_len, dp[i][j])

        # 注意满足的长度至少为3
        return 0 if max_len < 3 else max_len

    def sumSubarrayMins(self, A: List[int]) -> int:
        # 当前状态加dp，超时
        # st = [A[0]]
        # cur_min, lens = 30001, len(A)
        # dp = [A[0]] + [0] * (lens - 1)
        #
        # for i in range(1, lens):
        #     if A[i] < st[i-1]:
        #         for j in range(i):
        #             if A[i] < st[j]:
        #                 st[j] = A[i]
        #
        #     st.append(A[i])
        #     dp[i] = (dp[i - 1] + sum(st)) % (10 ** 9 + 7)
        #
        # return dp[-1]

        # 单调栈？？？（新玩意）
        # 思路：维护两个列表，一个表示左边第一个小于当前值的下标，
        # 一个维护右边第一个小于当前值的下标，
        # 最终收益为（i-left）*（right-i）*A[i]， 分别表示左右两边的选择数量（皆包含自身）
        # 总体思想：求取以A[i]为最小元素时的子列表的个数
        lens = len(A)
        left, right = [0] * lens, [0] * lens

        stack = []
        for i in range(lens):
            # 单调栈来保存过去的值所在下标，并与当前进行比较
            while stack and A[stack[-1]] > A[i]:
                stack.pop()

            if not stack:
                left[i] = -1
            else:
                # 栈中存在下标时，说明当前值左边第一个比其小的下标出现
                left[i] = stack[-1]
            stack.append(i)

        stack = []
        for i in range(lens - 1, -1, -1):
            # 对于相同的最小值，只需要取一次，因此只需使用一次严格大于
            while stack and A[stack[-1]] >= A[i]:
                stack.pop()

            if not stack:
                right[i] = lens
            else:
                right[i] = stack[-1]

            stack.append(i)

        return sum((i - left[i]) * (right[i] - i) * A[i] for i in range(lens)) % (10 ** 9 + 7)

    def partitionDisjoint(self, A: List[int]) -> int:
        # 不能只比较第一个元素,只想出了一半
        lens = len(A)
        # 记录从右到左的最小值列表以及从左到右的最大值列表
        left_max, right_min = [A[0]], [A[-1]]
        for i in range(1, lens):
            left_max.append(max(left_max[i - 1], A[i]))
            right_min.append(min(right_min[i - 1], A[lens - i - 1]))

        right_min.reverse()
        res = -1

        # 死活想不出这个判断条件
        for i in range(lens - 1):
            # 需要跳出相同位置带来的禁锢，因此判断的是前后的位置
            if right_min[i + 1] >= left_max[i]:
                res = i
                break

        return res + 1

    def maxSubarraySumCircular(self, A: List[int]) -> int:
        # 连续子数组和的方法不适用
        # res = A[0]
        # cur_len = 0
        # cur_sum = 0
        # lens = len(A)
        # A = A*2
        #
        # for value in A:
        #     if cur_sum > 0:
        #         cur_sum += value
        #         cur_len += 1
        #     else:
        #         cur_sum = value
        #         cur_len = 1
        #     if cur_len <= lens:
        #         res = max(cur_sum, res)
        #
        # return res

        # Kandane 算法：求取区间段[i,j]的最大子序列和，基本思想基于动态规划
        # dp[i] = A[i] + max(dp[i-1], 0)

        # 题目无非就是两个区间，对于单区间直接使用Kandane算法即可
        # 对于双区间最大，则采用逆向思维，如果双区间出现最大，则A[0],A[-1]必定包含在区间中
        # 因此可以转化为先求解区间[1, len-2]的最小连续子序列，再用整体和相减即可（神仙操作）
        lens = len(A)
        if lens < 2:
            return A[0]

        # 求解单区间max
        single_max = total = cur_sum = A[0]
        for i in range(1, lens):
            total += A[i]
            if cur_sum > 0:
                cur_sum += A[i]
            else:
                cur_sum = A[i]
            single_max = max(single_max, cur_sum)

        # 求解双区间min
        multi_min = cur_min = A[1]
        for i in range(2, lens - 1):
            if cur_min < 0:
                cur_min += A[i]
            else:
                cur_min = A[i]
            multi_min = min(multi_min, cur_min)

        # 最后比较两区间的max即可
        return max(single_max, total - multi_min)

    def minFlipsMonoIncr(self, S: str) -> int:
        # wrong case: 0101100011
        # 缺少对右边的判断
        # count_one, i = 0, 0
        # lens = len(S)
        # while i < lens and S[i] == '0':
        #     i += 1
        #
        # temp = i
        # while i < lens:
        #     if S[i] == '1':
        #         count_one += 1
        #     i += 1
        #
        # return min(lens-temp-count_one, count_one)

        # 拓展思路，去掉有序部分，再寻找最少的值
        # wrong:可以部分修改0，剩余的修改1 case "10011111110010111011"
        # lens = len(S)
        # left, right = 0, lens-1
        # while left < right and S[left] == '0':
        #     left += 1
        # while right > left and S[right] == '1':
        #     right -= 1
        #
        # count_zero = S[left:right+1].count('0')
        # return min(count_zero, right-left+1-count_zero)

        # 看来还是需要动态规划处理这些问题
        # 关键点：dp[i][0] 只能由 dp[i-1][0] 转变过来，因为需要满足递增的原则
        # 使用 dp[i][0] 来表示第i个位置为0时候的最小翻转次数
        # 由此有：当S[i] == '0' 时，有 dp[i][0] = dp[i-1][0], dp[i][1] = min(dp[i-1][0]+1, dp[i-1][1]+1)
        # 当S[i] == '1' 时，有 dp[i][0] = dp[i-1][0]+1, dp[i][1] = min(dp[i-1][0], dp[i-1][1])
        res_zero, res_one = 0, 0
        for s in S:
            if s == '0':
                res_one, res_zero = min(res_zero + 1, res_one + 1), res_zero
            else:
                res_zero, res_one = res_zero + 1, min(res_zero, res_one)

        return min(res_one, res_zero)

    def minIncrementForUnique(self, A: List[int]) -> int:
        # 贪心思想
        # 算法思想：先进行排序，然后与当前增序列逐一比对，
        # 若比序列值小，则说明需要进行操作
        A.sort()
        lens = len(A)
        if not lens:
            return 0
        # 算法思想有问题，案例：0,2,2
        # return sum(max(i-A[i], 0) for i in range(lens)) if not A[0] \
        #     else sum(max(i-A[i-1], 0) for i in range(1, lens+1))
        count = 0
        # cur_val保存的是邻近增量序列值
        cur_val = A[0]
        for val in A:
            temp = val - cur_val
            if temp < 0:
                count -= temp
            elif temp > 0:
                # 当val较大时，直接跳过去
                cur_val = val
            # 一般情况只增加一，使得操作次数最优
            cur_val += 1

        return count

    def deckRevealedIncreasing(self, deck: List[int]) -> List[int]:
        # 逆向思维很重要（秀的头皮发麻）
        from _collections import deque
        deck.sort()
        lens = len(deck)
        res = deque()
        for i in range(lens - 1, 0, -1):
            # 从尾到头重新推导
            res.append(deck[i])
            temp = res.popleft()
            res.append(temp)

        res.append(deck[0])
        # 将结果取逆即可
        return list(res)[::-1]

    def canReorderDoubled(self, A: List[int]) -> bool:
        # 基本思想：排序加分队列处理，使用字典加速查找
        dicts = {}
        A.sort()
        for e in A:
            if e in dicts:
                dicts[e] += 1
            else:
                dicts[e] = 1

        # 首先判断正数部分的队列，
        # 只要出现本身的为奇数，或者本身的一半不在字典以及本身的一半在字典的数量不足时，即为False
        cur_pos = len(A) - 1
        while cur_pos >= 0 and A[cur_pos] > 0:
            # 只要注意如果当前的值数量不够时直接跳过即可
            if dicts[A[cur_pos]] > 0:
                if A[cur_pos] % 2 == 0 and A[cur_pos] // 2 in dicts and \
                        dicts[A[cur_pos] // 2] > 0:
                    dicts[A[cur_pos]] -= 1
                    dicts[A[cur_pos] // 2] -= 1
                else:
                    return False
            cur_pos -= 1

        # 负数队列同理，只是顺序改为从左到右
        i = 0
        while i <= cur_pos:
            if dicts[A[i]] > 0:
                if A[i] % 2 == 0 and A[i] // 2 in dicts and \
                        dicts[A[i] // 2] > 0:
                    dicts[A[i]] -= 1
                    dicts[A[i] // 2] -= 1
                else:
                    return False
            i += 1

        # 当两边都能成功遍历时，说明满足题目条件
        return True

    def maxWidthRamp(self, A: List[int]) -> int:
        # 完全没思路
        # 方法一：排序法
        # 先对值进行排序，然后利用排序前的值所在的下标进行坡度长判断
        # res = 0
        # cur_min = 50001
        # # python的骚气写法，利用值作为排序的key来对下标进行排列（活到老学到老）
        # for i in sorted(range(len(A)), key=A.__getitem__):
        #     # 最大的坡度必为当前的下标减去先前的最小下标（因为排序使得值有序）
        #     res = max(res, i-cur_min)
        #     cur_min = min(cur_min, i)
        #
        # return res

        # 方法二：单调栈解法（还是没学会这个结构）
        # 维护一个单调栈，第一个值为列表中的第一个元素下标，最后一个值为列表中的最小值的下标
        # 从后向前遍历，出现比栈顶大的即进行一个坡度计算
        stack = []
        lens = len(A)
        # 单调栈保存的是下标，并且是值由大到最小的下标
        for i in range(lens):
            if not stack or A[i] <= A[stack[-1]]:
                stack.append(i)

        res = 0
        i = lens - 1
        # 这个结束条件很巧妙（值得学习）
        while i > res:
            # 当出现大于或等于栈中的值的时候，计算一次坡度，并保存最大值
            while stack and A[stack[-1]] <= A[i]:
                res = max(res, i - stack[-1])
                # 注意计算完成后需要弹出，因为说不定后面还有更大的坡度
                stack.pop()

            i -= 1

        return res

    def pancakeSort(self, A: List[int]) -> List[int]:
        # 思路：每次移动最大元素，并将其放入正确位置，不断循环直到最小元素
        # 引用了切片和内部索引，因此速度不是一般快
        # 不过时间复杂度为n2，空间复杂度为n（切片需要额外空间）
        lens = len(A)
        res = []
        for i in range(lens, 0, -1):
            cur_pos = A.index(i)
            if cur_pos != i - 1:
                # python部分列表反转的小技巧A[尾部元素:起始元素-1:-1]
                # 同样满足左闭右开原则
                A[:cur_pos + 1] = A[cur_pos::-1]
                # 如果只需要移动自身，则直接跳过
                if cur_pos != 0:
                    res.append(cur_pos + 1)
                A[:i] = A[i - 1::-1]
                res.append(i)

        return res

    def subarraysDivByK(self, A: List[int], K: int) -> int:
        # 还是太菜，知道需要用前缀和与hash表，但是忘了还有个同余定理
        # 并且根本不知道怎么把这些联系起来（难于上青天）
        from collections import Counter
        p = [0]
        for x in A:
            # 添加同余后是结果
            p.append((p[-1] + x) % K)

        print(p)
        c = Counter(p).values()
        # 同余定理：如果有两个整数满足 (a-b)%k == 0, 则有 a%k == b%k
        # 因此最终只需要利用组合计算前缀和相等的数量
        return sum(v * (v - 1) // 2 for v in c)

    def maxTurbulenceSize(self, A: List[int]) -> int:
        # 感觉贪心就可以了,没必要使用动态规划
        # 添加相同尾部，减少尾部判别条件
        A.append(A[-1])
        lens = len(A)

        # 使用符号标记过去的符号，1表示大于，0表示小于
        pre_sym = -1
        count, tot = 1, 0
        for i in range(1, lens):
            # 出现不同元素才进行添加
            if A[i] != A[i - 1]:
                sym = 0
                if A[i] < A[i - 1]:
                    sym = 1

                if pre_sym == -1 or sym != pre_sym:
                    count += 1
                else:
                    # 如果不再满足条件，则计算结果
                    tot = max(tot, count)
                    count = 2
                pre_sym = sym
            else:
                # 如果出现相同元素，则直接计算结果，并重置各项变量
                tot = max(tot, count)
                count, pre_sym = 1, -1

        return tot

    def minDominoRotations(self, A: List[int], B: List[int]) -> int:
        # 基本思想：先找出出现最多的元素，然后再对该元素的出现情况进行遍历，
        # 发现不满足的条件即退出，核心也是贪心思想，时间复杂度为O(n)（虽然需要遍历两遍）
        count_a, count_b = [0] * 6, [0] * 6
        for a, b in zip(A, B):
            count_a[a - 1] += 1
            count_b[b - 1] += 1

        lens = len(A)
        # 寻找出现最多的元素
        max_v, value = count_a[0] + count_b[0], 0
        for i in range(1, 6):
            temp = count_a[i] + count_b[i]
            if temp > max_v:
                max_v, value = temp, i

        # 如果出现最多的不能满足最低条件，直接返回
        if max_v < lens:
            return -1
        else:
            for index in range(lens):
                if A[index] != value + 1 and B[index] != value + 1:
                    return -1

            return lens - max(count_b[value], count_a[value])

    def shipWithinDays(self, weights: List[int], D: int) -> int:
        # 逆向思维很重要，
        # 将过程进行模拟，约定上下界即可（这个思维转化确实值得学习）
        # 过程模拟函数
        def getDay(w: int) -> int:
            count = 0
            cur = 0
            for wei in weights:
                cur += wei
                # 对于多出来的后面会继续运行，因此无需添加额外判断
                if cur > w:
                    cur = wei
                    count += 1

            # 若最终cur为0时，则无需增加计数
            return count + 1 if cur else count

        # 范围必为最大值与总和之间
        right = sum(weights)
        left = max(weights)
        # 然后二分查找即可
        while left < right:
            mid = left + (right - left) // 2
            days = getDay(mid)
            if days > D:
                left = mid + 1
            else:
                right = mid

        return left

    def maxScoreSightseeingPair(self, A: List[int]) -> int:
        # 有动态规划的思想，将整体划分为 A[i]+i 与 A[j]-j 两个部分
        # 先求取A[i]+i列表，再利用DP来求取A[j]-j列表中当前长度的最大值
        # 需要进行三次遍历，时间复杂度为O（n）
        # lens = len(A)
        # # A[i]+i
        # sort_i = [index+A[index] for index in range(lens-1)]
        # # A[j]-j 中当前长度的最大值
        # sort_j = [A[lens-1]-lens+1]
        # for i in range(lens-2, 0, -1):
        #     temp = A[i] - i
        #     if temp > sort_j[-1]:
        #         sort_j.append(temp)
        #     else:
        #         sort_j.append(sort_j[-1])
        #
        # cur_max = 0
        # # 最后一次遍历求取整体max
        # for i in range(lens-1):
        #     cur_max = max(cur_max, sort_i[i]+sort_j[lens-2-i])
        #
        # return cur_max

        # 网上大神思路：只需要一次遍历
        # 关键点：两个所需要计算的变量都是固定的，
        # 并且位置也固定，因此只需要逐一遍历即可
        # left 保存先前的max A[i]+i
        # A[j]-j 则直接进行遍历
        left, res, lens = A[0], 0, len(A)
        for j in range(1, lens):
            res = max(res, left + A[j] - j)
            left = max(left, A[j] + j)

        return res

    def maxSumTwoNoOverlap(self, A: List[int], L: int, M: int) -> int:
        # 最直白的方法：设置四个列表分别保存左边部分与右边部分L，M的max
        # wrong: 未知名错误，思路应该是没错，就是下标太烦人了，懒得改了
        # lens, pre_sum = len(A), [0]
        # for i in range(lens):
        #     pre_sum.append(pre_sum[i]+A[i])
        #
        # left_l, left_m, right_l, right_m = [0]*lens, [0]*lens, [0]*lens, [0]*lens
        # for i in range(L-1, lens):
        #     left_l[i] = max(left_l[i-1], pre_sum[i+1]-pre_sum[i-L+1])
        #     right_l[i] = max(right_l[i-1], pre_sum[lens+L-i-1]-pre_sum[lens-i-1])
        # right_l.reverse()
        #
        # for i in range(M-1, lens):
        #     left_m[i] = max(left_m[i-1], pre_sum[i+1]-pre_sum[i-M+1])
        #     right_m[i] = max(right_m[i-1], pre_sum[lens+M-i-1]-pre_sum[lens-i-1])
        # right_m.reverse()
        #
        # max_sum = 0
        # for i in range(lens):
        #     max_sum = max(max(left_l[i]+right_m[i], left_m[i]+right_l[i]), max_sum)
        #
        # return max_sum

        # 重新尝试一下
        lens = len(A)
        pre_sum = [0]
        # 利用动态规划的思想保存各值，避免二次计算
        # 分别设置四个列表保存各个方向的两个最大子列表的和
        # 注意左侧不计算i位的值，右侧需要计算i位的值
        # 并且考虑到步长1的原因，右侧需要额外一个空间存储初始位，防止溢出
        left_l, left_m, right_l, right_m = [0] * lens, [0] * lens, [0] * (lens + 1), [0] * (lens + 1)

        # 前缀和
        for i in range(lens):
            pre_sum.append(pre_sum[i] + A[i])

        # 左侧L值
        for i in range(L, lens):
            left_l[i] = max(left_l[i - 1], pre_sum[i] - pre_sum[i - L])
        # 右侧L值
        for i in range(lens - L, -1, -1):
            right_l[i] = max(right_l[i + 1], pre_sum[i + L] - pre_sum[i])
        # 左侧M值
        for i in range(M, lens):
            left_m[i] = max(left_m[i - 1], pre_sum[i] - pre_sum[i - M])
        # 右侧M值
        for i in range(lens - M, -1, -1):
            right_m[i] = max(right_m[i + 1], pre_sum[i + M] - pre_sum[i])

        res = 0
        # 最后进行两边判断即可
        for i in range(lens):
            res = max(res, max(left_l[i] + right_m[i], left_m[i] + right_l[i]))

        return res

    def maxUncrossedLines(self, A: List[int], B: List[int]) -> int:
        # 动态规划状态转移方程死活想不出来，还是学不会
        # 实质为求取最长公共子序列（不需要相连）
        len_a, len_b = len(A), len(B)
        dp = [[0] * (len_b + 1) for _ in range(len_a + 1)]

        for i in range(len_a):
            for j in range(len_b):
                # 相等时，则为前一个的值加上一
                if A[i] == B[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                # 不等时，为之前的最大值
                else:
                    dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

        return dp[len_a][len_b]

    def numMovesStonesII(self, stones: List[int]) -> List[int]:
        # 思路确实很难想到，尤其是最大值部分的有关s2的求解
        # 最小值的话找出初始时已经摆好的最多的石子，在实现上有难度（需要用到滑动窗口的思想）
        stones.sort()
        lens = len(stones)
        # s1为总的空位
        s1 = stones[lens - 1] - stones[0] + 1 - lens
        # s2为移动左右端点后所减去的空位，由于取的是最大值，因此选最小
        s2 = min(stones[1] - stones[0] - 1, stones[lens - 1] - stones[lens - 2] - 1)
        mx = s1 - s2
        mi, j = mx, 0
        for i in range(lens):
            # 当前窗口比元素总数高时，缩小窗口
            while j + 1 < lens and stones[j + 1] - stones[i] + 1 <= lens:
                j += 1
            # 当前完成移动所需的开销，当前窗口的石头数量为 j-i+1
            cost = lens - (j - i + 1)
            # 如果出现只有最后一个错位时，还是需要两次搬运：2,3,4,7 或者 1,6,7,8
            # 判断条件很灵性，不用考虑位置颠倒
            if j - i + 1 == lens - 1 and stones[j] - stones[i] + 1 == lens - 1:
                cost = 2
            mi = min(mi, cost)

        return [mi, mx]

    def maxSatisfied(self, customers: List[int], grumpy: List[int], X: int) -> int:
        # 有使用到滑动窗口以及贪心思想，但是窗口长度固定，所以还比较好写
        lens = len(customers)
        # cur_sum 维护一个长度为X的列表的和，其中只包含grumpy为1时的值
        total, cur_sum, max_sum = 0, 0, 0
        for i in range(lens):
            if not grumpy[i]:
                total += customers[i]
            else:
                cur_sum += customers[i]

            # 当且仅当窗口末端的值的grumpy为1时，才进行剔除
            if i >= X and grumpy[i - X]:
                cur_sum -= customers[i - X]

            # 取仅含grumpy为1的最大值
            max_sum = max(max_sum, cur_sum)

        # 最后将0与1的值进行汇总即可
        return max_sum + total

    def prevPermOpt1(self, A: List[int]) -> List[int]:
        # 有用到贪心的思想
        lens = len(A)
        switch, i = -1, 0
        if not switch:
            return A

        # 从右向左寻找第一个非升序元素
        for i in range(lens - 2, -1, -1):
            if A[i] > A[i + 1]:
                switch = i
                break
        # 如果当前序列已经全部有序，则必为最小序列，直接进行返回
        if switch < 0:
            return A

        i = lens - 1
        # 然后从右向左寻找小于非升序元素的最靠左边的位置，再进行交换
        # 没有考虑到的问题，右边最大元素可能大于等于左边交换位置元素
        while A[i] >= A[switch] or A[i] == A[i - 1]:
            i -= 1

        A[i], A[switch] = A[switch], A[i]
        return A

    def corpFlightBookings(self, bookings: List[List[int]], n: int) -> List[int]:
        # 暴力法：超时毋庸置疑
        # res = [0]*n
        #
        # for book in bookings:
        #     for i in range(book[0]-1, book[1]):
        #         res[i] += book[2]
        #
        # return res

        # 区间累加可以考虑使用差分数列（活到老学到老）
        # 优化的关键点，将内层循环优化为O(1)
        # 大神思想：定义一个差分列表，保存当前人数与上一个的人数之差
        # 则有d[i] = res[i]-res[i-1]
        # 最后进行累加即可（这个打死都想不出来）
        diff, res = [0] * n, [0] * n
        # 想象成公交车站的上下车问题，i处上车，j+1处下车
        for book in bookings:
            diff[book[0] - 1] += book[2]
            if book[1] < n:
                diff[book[1]] -= book[2]

        res[0] = diff[0]
        # 最后的累加由此处的公式求解
        for i in range(1, n):
            res[i] = res[i - 1] + diff[i]

        return res

    def movesToMakeZigzag(self, nums: List[int]) -> int:
        # wrong:没有动态对列表进行修改
        # low_cost, high_cost = 0, 0
        # lens = len(nums)
        # if lens < 2:
        #     return 0
        #
        # for i in range(1, lens):
        #     if nums[i] > nums[i-1]:
        #         if i % 2:
        #             high_cost += nums[i] - nums[i-1] + 1
        #         else:
        #             low_cost += nums[i] - nums[i-1] + 1
        #     elif nums[i] < nums[i-1]:
        #         if i % 2:
        #             low_cost += nums[i-1] - nums[i] + 1
        #         else:
        #             high_cost += nums[i-1] - nums[i] + 1
        #     else:
        #         low_cost += 1
        #         high_cost += 1
        #
        # return min(low_cost, high_cost)

        # 主要思想：贪心思想，然后直接对两种情况进行模拟，最后取其中的小值即可
        # 时间复杂度为O(n)，空间复杂度为O(n)
        # 错误原因，没有读题：只能减不能增加
        # 由此带来的区别是，
        # 当初始为递增序列时，如果在偶数位遇到较小值，则对前一位进行修改
        # 由于后序序列的计算不需要上上个元素的值，因此，就不需要对当前进行修改
        # 反之对于递减也是如此
        lens = len(nums)
        low, high = 0, 0
        arr = [nums[0]] + [0] * (lens - 1)
        for i in range(1, lens):
            cur_val = nums[i]
            if i % 2 and cur_val >= arr[i - 1]:
                high += cur_val - arr[i - 1] + 1
                cur_val = arr[i - 1] - 1
            elif not i % 2 and cur_val <= arr[i - 1]:
                high += arr[i - 1] - cur_val + 1
                # 此处应该添加对i-1位置的修改，但是后面不需要用到，就直接省略了
            arr[i] = cur_val

        for i in range(1, lens):
            cur_val = nums[i]
            if i % 2 and cur_val <= arr[i - 1]:
                # 同理此处也是如此
                low += arr[i - 1] - cur_val + 1
            elif not i % 2 and cur_val >= arr[i - 1]:
                low += cur_val - arr[i - 1] + 1
                cur_val = arr[i - 1] - 1

            arr[i] = cur_val

        return min(low, high)

    def invalidTransactions(self, transactions: List[str]) -> List[str]:
        # 思路有问题？？
        # tran = []
        # for s in transactions:
        #     tran.append(s.split(','))
        # tran.sort(key=lambda x: [x[0], int(x[1])])
        # lens = len(tran)
        # res = []
        # if int(tran[0][2]) >= 1000:
        #     res.append(','.join(tran[0]))
        #
        # for i in range(1, lens):
        #     flag = False
        #
        #     if int(tran[i][2]) >= 1000:
        #         flag = True
        #     if tran[i][0] == tran[i - 1][0] and \
        #             tran[i][3] != tran[i - 1][3] and \
        #             int(tran[i][1]) - int(tran[i - 1][1]) <= 60:
        #         if int(tran[i - 1][2]) <= 1000:
        #             res.append(','.join(tran[i - 1]))
        #         flag = True
        #
        #     if flag:
        #         res.append(','.join(tran[i]))
        #
        # return res

        # 暴力法简单粗暴
        # 时间复杂度为O(n2)
        # 对每一个都进行从头到尾的遍历判断，满足条件即加入
        tran = [x.split(',') for x in transactions]
        res = []

        # 遍历每一个列表
        for i, st in enumerate(tran):
            if int(st[2]) > 1000:
                res.append(','.join(st))
                continue
            # 依次遍历列表，查看是否有能与当前列表满足条件的列表
            for j, ed in enumerate(tran):
                if i == j:
                    continue
                if st[0] == ed[0] and st[3] != ed[3] and abs(int(st[1]) - int(ed[1])) <= 60:
                    res.append(','.join(st))
                    break

        return res

    def canMakePaliQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        # 思想很有意思：尤其是当k>=13这一点很有趣
        # 需要注意子串是可以重排的，因此只需要看各字母出现奇数次的个数即可
        # 虽然有点巧妙，但是还是暴力法，因此无法通过最后几个变态长度的案例
        # from collections import Counter
        # res = [True] * len(queries)
        # for i, val in enumerate(queries):
        #     if val[2] < 13:
        #         tmp = Counter(s[val[0]:val[1] + 1])
        #         odd = 0
        #         for cur in tmp.values():
        #             if cur % 2:
        #                 odd += 1
        #         # 如果弥补的数量少于能容许的数量，则为False
        #         if odd - 2 * val[2] > (val[1] - val[0] + 1) % 2:
        #             res[i] = False
        #
        # return res

        # 尝试动态规划保存当前的字符出现次数
        lens = len(s)
        dp = [[0] * 26 for _ in range(lens + 1)]

        # 统计到长度为i时的各字符出现次数
        for i in range(len(s)):
            # py浅拷贝列表元素
            dp[i + 1][:] = dp[i][:]
            # 每次只需要增加新增元素即可
            dp[i + 1][ord(s[i]) - ord('a')] += 1

        res = [True] * len(queries)
        for index, val in enumerate(queries):
            odd = 0
            for i in range(26):
                # dp[right+1]-dp[left]即为长度区间在[left, right]的元素分布
                odd += (dp[val[1] + 1][i] - dp[val[0]][i]) % 2

            # 判断剩余数量满足当前子串长度要求，奇数长度可以允许多出一个元素
            if odd - 2 * val[2] > (val[1] - val[0] + 1) % 2:
                res[index] = False

        return res

    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        # 并查集，有点忘了怎么实现了..
        # 并查集返回的好像也不是最终结果
        # lens, count = len(s), 0
        # belong = [i for i in range(lens)]
        #
        # # 并查集算法
        # def find(root):
        #     if belong[root] != root:
        #         belong[root] = find(belong[root])
        #     return belong[root]
        #
        # # 需要好好斟酌
        # for left, right in pairs:
        #     # pa_left, pa_right = find(left), find(right)
        #     # if pa_left != pa_right:
        #     #     belong[pa_left] = pa_right
        #     # union 操作可以直接合并为一句，牛皮
        #     belong[find(left)] = find(right)
        #
        # # 排序
        # dicts = {}
        # for i in range(lens):
        #     if belong[i] not in dicts:
        #         dicts[belong[i]] = ''
        #     dicts[belong[i]] += s[i]
        #
        # for key in dicts:
        #     dicts[key] = sorted(dicts[key])
        #
        # res = ['']*lens
        # for key in dicts:
        #     sort = dicts[key]
        #     k = 0
        #     for i in range(lens):
        #         if belong[i] == key:
        #             res[i] = sort[k]
        #             k += 1
        #
        # return ''.join(res)

        # 并查集加对序列排序放入
        # 亮点，使用字典存储所属顶层节点的所有下标而非具体的字符
        # 便于后续的放回处理
        import collections
        p = {i: i for i in range(len(s))}  # 初始化并查集

        def f(x):
            # 包含了路径压缩的find算法
            if x != p[x]:
                p[x] = f(p[x])
            return p[x]

        for i, j in pairs:
            # union操作
            p[f(j)] = f(i)

        d = collections.defaultdict(list)
        # 对并查集中的每一个元素都再使用find函数找出其顶层节点的值
        for i, j in enumerate(map(f, p)):
            # 将不同元素加入所属的顶层节点
            d[j].append(i)
        # 排序
        ans = list(s)
        # 每个顶层节点所包含的元素集合
        for q in d.values():
            # 对元素所指代的字符进行排序
            t = sorted(ans[i] for i in q)
            # 将元素与其指代值解压放入
            for i, c in zip(sorted(q), t):
                ans[i] = c
        return ''.join(ans)

    def equalSubstring(self, s: str, t: str, maxCost: int) -> int:
        # 题目都没交代清楚：应该是转化为最长子串，而非子序列
        # 必须连续，因此可以采用滑动窗口来解决
        lens = len(s)
        res = [abs(ord(s[i]) - ord(t[i])) for i in range(lens)]
        # i,j 分别为窗口的左右边界，一旦长度超过，则移动左指针，其余都是移动右指针
        j, tot, max_len = 0, 0, 0
        for i in range(lens):
            tot += res[i]
            if tot > maxCost:
                tot -= res[j]
                j += 1
            max_len = max(max_len, i - j + 1)

        return max_len

    def queensAttacktheKing(self, queens: List[List[int]], king: List[int]) -> List[List[int]]:
        # 预设基准，减少条件判断语句
        judge = [[0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1]]
        # 由于每一个queen都需要进行最多8次计算，因此还是比较耗时的
        res = [[judge[i][0] * 100, judge[i][1] * 100] for i in range(8)]

        for x, y in queens:
            diffx, diffy = x - king[0], y - king[1]
            for i in range(8):
                # 通过判断两坐标与基准的比值是否相等以及比值大小来进行放置
                if judge[i][0]:
                    temp = diffx // judge[i][0]
                    if 0 < temp < (res[i][0] - king[0]) // judge[i][0] and temp * judge[i][1] == diffy:
                        res[i] = [x, y]
                        break

                # 判断纵坐标
                if judge[i][1]:
                    temp = diffy // judge[i][1]
                    if temp * judge[i][0] == diffx and 0 < temp < (res[i][1] - king[1]) // judge[i][1]:
                        res[i] = [x, y]
                        break

        # 过滤不在比较范围内的位置
        return [x for x in res if abs(x[0]) != 100 and abs(x[1]) != 100]

    def removeSubfolders(self, folder: List[str]) -> List[str]:
        # 由于需要使用排序，因此时间复杂度为O(nlogn)，空间复杂度为O(1)，不含结果列表
        folder.sort()
        # 添加尾部结束符，减少多余的判断
        folder.append('/')
        sht_str, sht_cut = folder[0], folder[0].split('/')
        res = []
        for cur_str in folder[1:]:
            cur_cut = cur_str.split('/')
            sht_len = min(len(cur_cut), len(sht_cut)) - 1
            # 当两者都不同时，可以直接添加，
            # 否则就是出现诸如 /a/ab 与 /a/abc 等的情况，此时需要判断较短串尾部的字符是否相等
            if sht_str not in cur_str or cur_cut[sht_len] != sht_cut[sht_len]:
                res.append(sht_str)
                sht_str = cur_str
                sht_cut = sht_str.split('/')

        return res

    def countServers(self, grid: List[List[int]]) -> int:
        # 思路一：常规法
        # # 基本思路：使用行/列列表来分别统计每行/列出现的次数,然后根据统计的值来判断每个位置是否满足
        # row, col = len(grid), len(grid[0])
        # count_row, count_col = [0] * row, [0] * col
        # for i in range(row):
        #     for j in range(col):
        #         if grid[i][j]:
        #             count_row[i] += 1
        #             count_col[j] += 1
        #
        # # res = 0
        # # for i in range(row):
        # #     for j in range(col):
        # #         # 这里很巧妙，只要行列中有一个出现次数超过1时，说明相互有连接
        # #         res += grid[i][j] and (count_row[i]>1 or count_col[j]>1)
        # # return res
        #
        # return sum(grid[i][j] and (count_row[i] > 1 or count_col[j] > 1) for i in range(row) for j in range(col))

        # 思路二：并查集
        from collections import defaultdict
        row, col = len(grid), len(grid[0])
        pre = [i for i in range(row + col)]

        # 并查集的基本操作
        # 寻找同一集合的元素
        def find(root: int):
            if pre[root] != root:
                pre[root] = find(pre[root])
            return pre[root]

        # 合并元素
        def union(x: int, y: int):
            pre[find(x)] = find(y)

        # 先统计点总数
        total = 0
        for i in range(row):
            for j in range(col):
                if grid[i][j]:
                    total += 1
                    # 这里没看懂
                    union(i, j + row)

        # 统计每个集合的元素数
        dparent = defaultdict(int)
        for i in range(row):
            for j in range(col):
                if grid[i][j]:
                    dparent[find(i)] += 1

        # 减去只包含一个元素的集合个数
        return total - sum(1 for i in dparent.values() if i == 1)

    def countSquares(self, matrix: List[List[int]]) -> int:
        # 纯粹的暴力法（碰到满足的顺序往下迭代）
        count, row, col = 0, len(matrix), len(matrix[0])
        for i in range(row):
            for j in range(col):
                if matrix[i][j]:
                    count += 1
                    ext_row, ext_col = i + 1, j + 1
                    while ext_row < row and ext_col < col and matrix[i][ext_col] == matrix[ext_row][j] == 1:
                        k = 1
                        while i + k <= ext_row and matrix[i + k][ext_col] == matrix[ext_row][j + k] == 1:
                            k += 1
                        # 出现隐秘的错误：当较小的正方形都不满足时，较大的肯定也不满足
                        if i + k > ext_row:
                            count += 1
                            ext_row, ext_col = ext_row + 1, ext_col + 1
                        else:
                            break
        return count

    def maxSideLength(self, mat: List[List[int]], threshold: int) -> int:
        # 还是需要使用前缀和来解决，虽然比较耗时
        row, col = len(mat), len(mat[0])
        pre_sum = [[0] * (col + 1) for _ in range(row + 1)]
        for i in range(row):
            for j in range(col):
                # 构建二维前缀和
                pre_sum[i + 1][j + 1] = mat[i][j] + pre_sum[i][j + 1] + pre_sum[i + 1][j] - pre_sum[i][j]

        tot_min = 0
        # 直接暴力判断会超时，需要进行二分
        # # 细节真心太差了，错误点：行列总数
        # for k in range(1, min(row, col)+1):
        #     cur_min = 200000
        #     for i in range(k, row+1):
        #         for j in range(k, col+1):
        #             cur_min = min(cur_min, pre_sum[i][j]-pre_sum[i][j-k]-pre_sum[i-k][j]+pre_sum[i-k][j-k])
        #
        #     # 错误点：门限可以相等
        #     if cur_min <= threshold:
        #         tot_min = k

        # 使用二分来进行查找k值，优化时间复杂度
        left, right = 1, min(row, col) + 1
        while left < right:
            k = left + (right - left) // 2

            cur_min = 200000
            for i in range(k, row + 1):
                for j in range(k, col + 1):
                    # 实质为前缀和列表的逆运算
                    cur_min = min(cur_min,
                                  pre_sum[i][j] - pre_sum[i][j - k] - pre_sum[i - k][j] + pre_sum[i - k][j - k])

            # 利用计算的门限来判断收缩的边界
            if cur_min > threshold:
                right = k
            else:
                tot_min = k
                left = k + 1

        return tot_min

    def isPossibleDivide(self, nums: List[int], k: int) -> bool:
        # 暴力法：超时，还是不得行。
        # 基本思路：将已经判断过的数字置零，最终求和计算结果
        # lens = len(nums)
        # if lens % k:
        #     return False
        # nums.sort()
        #
        # i = 0
        # while i < lens:
        #     if nums[i]:
        #         j = i
        #         cur_val = nums[j] - 1
        #
        #         for _ in range(k):
        #             # 暴力穷举
        #             while j < lens and nums[j] < cur_val + 1:
        #                 j += 1
        #             if j == lens or nums[j] != cur_val + 1:
        #                 return False
        #
        #             cur_val += 1
        #             nums[j] = 0
        #             j += 1
        #
        #     i += 1
        #
        # return True if not sum(nums) else False

        # 思路二，使用hash排序后进行模拟遍历
        # 主要优势点：将重复组直接过滤，比逐一遍历过滤确实要快很多
        from collections import Counter
        count = Counter(nums)
        sort_arr = sorted(count)
        for num in sort_arr:
            time = count[num]
            if time > 0:
                for i in range(num, num + k):
                    # 直接减掉所有重复组
                    if count[i] >= time:
                        count[i] -= time
                    else:
                        return False

        return True

    def findBestValue(self, arr: List[int], target: int) -> int:
        # 问题很大，需要修改方式，直接暴力也可能会超时
        # arr.sort()
        # lens = len(arr)
        # res, pre_val = round(target/lens), 0
        # min_mod = target % lens
        # if res > arr[0]:
        #     for i in range(lens-1):
        #         target -= arr[i]
        #         if arr[i] != pre_val and arr[i] >= round(target/(lens-1-i)) and min_mod > target % (lens-1-i):
        #             min_mod = target % (lens-1-i)
        #             res = round(target / (lens-1-i))
        #         pre_val = arr[i]
        #
        # return res
        # 需要仔细斟酌
        # 可以用这个思路，但是基本思想有问题
        arr.sort()
        cur_sum, lens = 0, len(arr)
        for i in range(lens):
            # 每次减去当前已经遍历过的总和
            cur_avg = (target - cur_sum) // (lens - i)
            # 这个结束条件没有想到
            # 出现当前的avg值小于等于当前的值时，即可返回
            if cur_avg <= arr[i]:
                return round((target - cur_sum) / (lens - i))

            cur_sum += arr[i]

        return arr[-1]

    def diagonalSort(self, mat: List[List[int]]) -> List[List[int]]:
        # 最朴素的方法：逐一遍历，再排序放回
        # 代码有点冗余
        row, col = len(mat), len(mat[0])
        if not row:
            return mat

        def get_sort(crow, ccol):
            arr_sort = []
            while 0 <= crow < row and 0 <= ccol < col:
                arr.append(mat[crow][ccol])
                crow += 1
                ccol += 1

            arr_sort.sort()
            return arr_sort

        # 左下角部分
        for i in range(row - 1, -1, -1):
            cur_row, cur_col = i, 0
            arr = get_sort(cur_row, cur_col)

            temp_col = 0
            for j in range(i, cur_row):
                mat[j][temp_col] = arr[j - i]
                temp_col += 1

        # 右上角部分
        for i in range(1, col):
            cur_row, cur_col = 0, i
            arr = get_sort(cur_row, cur_col)

            temp_row = 0
            for j in range(i, cur_col):
                mat[temp_row][j] = arr[j - i]
                temp_row += 1

        return mat

    def filterRestaurants(self, restaurants: List[List[int]], veganFriendly: int, maxPrice: int, maxDistance: int) -> \
            List[int]:
        # 插入排序，每次找到满足条件的进行顺序插入
        # 错误一：id号并不与当前位置存在关系，需要建立映射
        # res = []
        #
        # if not restaurants:
        #     return res
        # relas = {}
        # # 建立id与餐厅的映射
        # for i in restaurants:
        #     relas[i[0]] = i[1:]
        #
        # for rest in restaurants:
        #     # 对于满足条件的，采用简单插入排序来处理
        #     if (not veganFriendly or rest[2]) and rest[3] <= maxPrice and rest[4] <= maxDistance:
        #         # 时间复杂度为O(n)，尝试简化
        #         # j = 0
        #         # while j < len(res) and (
        #         #         relas[res[j]][0] > restaurants[i][1] or relas[res[j]][0] == restaurants[i][
        #         #     1] and res[j] > restaurants[i][0]):
        #         #     j += 1
        #         # res.insert(j, restaurants[i][0])
        #         res.append(rest[0])
        #
        # # 内置函数是真滴牛皮，直接提升了一倍速度
        # res.sort(key=lambda x: (-relas[x][0], -x))
        #
        # return res

        # 简化版: 列表推导+内置多重排序
        # 根本不需要hash来建立id与对应的餐馆的映射
        if not restaurants:
            return []
        res = [rest for rest in restaurants if
               (not veganFriendly or rest[2]) and rest[3] <= maxPrice and rest[4] <= maxDistance]
        res.sort(key=lambda x: (-x[1], -x[0]))
        return [x[0] for x in res]

    def minSetSize(self, arr: List[int]) -> int:
        # 基本思想是建立一个hash映射，关联出现的数字以及其出现的次数
        from collections import Counter
        relas = Counter(arr)
        lens, count = len(arr), 0
        # 对其进行排序再从大到小挑选，有贪心思想
        choose = sorted(relas.keys(), key=lambda x: -relas[x])
        for i in range(len(choose)):
            count += relas[choose[i]]
            if count >= lens // 2:
                return i + 1

        # 按照题给条件时不会跳到这一步的
        return 0

    def numOfSubarrays(self, arr: List[int], k: int, threshold: int) -> int:
        # 尝试使用回溯，日，子数组
        # lens = len(arr)
        # if not lens:
        #     return 0
        #
        # path = []
        #
        # def travel(index: int):
        #     if len(path) == k:
        #         return 1 if sum(path) // k >= threshold else 0
        #
        #     count = 0
        #     for i in range(index, lens):
        #         path.append(arr[i])
        #
        #         count += travel(i + 1)
        #         path.pop()
        #
        #     return count
        #
        # return travel(0)

        # 那应该用前缀和就行，时间复杂度为O(n)，空间复杂度为O(1)，尝试优化为O(1)
        lens = len(arr)
        if not lens:
            return 0

        # pre_sum = [0]*(lens+1)
        #
        # for i in range(1, lens+1):
        #     pre_sum[i] = pre_sum[i-1]+arr[i-1]
        #
        # return sum(1 for i in range(k, lens+1) if (pre_sum[i]-pre_sum[i-k])//k >= threshold)

        # 将空间复杂度降为O(1)，并且只需要进行一次遍历
        count, cur_sum = 0, 0
        for i in range(lens):
            cur_sum += arr[i]
            if i >= k:
                cur_sum -= arr[i - k]

            # 能不用除法一定不要使用除法
            count += i >= k - 1 and cur_sum >= threshold * k

        return count

    def rankTeams(self, votes: List[str]) -> str:
        # 尝试转化为权重值,虽然可行，但是权重已经上升到了指数级别
        # 为了使得前一名的票数为绝对优势所带来的开销有点大
        # 时间复杂度为O(n2)，空间复杂度为O(m2)
        # value = [0]*26
        # max_count = 50
        #
        # for vote in votes:
        #     lens = len(vote)
        #     for i in range(lens):
        #         value[ord(vote[i])-ord('A')] += 10**(max_count-2*i)
        #
        # # 利用列表中的值对列表进行排序，
        # return "".join(chr(i[0]+ord('A')) for i in sorted(enumerate(value), key=lambda x: (-x[1], x[0])) if i[1] > 0)

        # 思路二：直接撸起袖子干，不搞花里胡哨的
        # 这个lambda函数有点优秀啊，会对列表自动进行顺序排序
        # 相比于上一个思路，直接提升了一倍性能可还行
        dicts = {}

        lens = len(votes[0])
        # 先逐一保存键值对
        for vote in votes:
            for i in range(lens):
                if vote[i] not in dicts:
                    dicts[vote[i]] = [0] * 26
                dicts[vote[i]][i] += 1

        # 再以多重排序来得到最终结果
        # 因为需要使用列表作为判断依据，因此不好直接在其顺序上做处理，因此需要使用reverse参数
        # 需要注意的是字符并不能像数字一样直接取反表示逆方向，因此需要转化
        return ''.join(sorted(dicts, key=lambda x: (dicts[x], -ord(x)), reverse=True))

    def numTimesAllBlue(self, light: List[int]) -> int:
        # 暴力法，O(n2) 超时
        # count, lens = 0, len(light)
        # visit = [0] * lens
        # max_pos = light[0]-1
        #
        # for pos in light:
        #     visit[pos - 1] = 1
        #     max_pos = max(max_pos, pos)
        #     if visit[0]:
        #         temp = max_pos-1
        #         # 在遇到黄灯时需要逐一向前遍历，此处应该能优化
        #         while temp and visit[temp]:
        #             temp -= 1
        #         if temp == 0 or visit[temp] == 2:
        #             visit[max_pos - 1] = 2
        #             count += 1
        #
        # return count

        # 还是没有理解透彻题目的内容：
        # 关键点：如果点亮的灯的数量与当前最远的灯的位置相等时，说明此时状态满足所有的灯为蓝色
        # 如果为大于时，说明至少存在一个黄灯（最远的那个），因此可以变成一次遍历
        # 时间复杂度为O(n)
        count, lens = 0, len(light)
        max_pos = -1
        for i in range(lens):
            max_pos = max(max_pos, light[i] - 1)
            # 只需要进行简单的条件判断即可
            if max_pos == i:
                count += 1

        return count

    def maxNumberOfFamilies(self, n: int, reservedSeats: List[List[int]]) -> int:
        # 一个很憨批的暴力法，逐一遍历，并且包含各种细节判断
        # reservedSeats.sort(key=lambda x: (x[0], x[1]))
        # lens = len(reservedSeats)
        # i, j, count = 1, 0, 0
        # while i < n + 1:
        #     # 出现j中的最大长度未达到n时，直接自增并结束语句
        #     if j == lens:
        #         count += (n - reservedSeats[j - 1][0]) * 2
        #         break
        #     # 当j中的长度已经超过i时，直接自增
        #     if i < reservedSeats[j][0]:
        #         count += 2 * (reservedSeats[j][0] - i)
        #         i = reservedSeats[j][0]
        #
        #     pre_dis = cur_max = 0
        #     while j < lens and reservedSeats[j][0] == i:
        #         # 考虑当前只出现5的情况
        #         if reservedSeats[j][1] == 5:
        #             pre_dis = 1
        #         # 计算左边的剩余
        #         count += (reservedSeats[j][1] - pre_dis - 1) // 4
        #         # 单独针对只出现一次9，出现2，7，以及出现4,9的特殊情况
        #         if (not pre_dis and reservedSeats[j][1] == 9) or (pre_dis == 2 and reservedSeats[j][1] == 7) or (
        #                 pre_dis == 4 and reservedSeats[j][1] == 9):
        #             count -= 1
        #         pre_dis = cur_max = reservedSeats[j][1]
        #         j += 1
        #     # 计算右边的剩余
        #     if cur_max < 6:
        #         count += (10 - cur_max - 1) // 4
        #     i += 1
        #
        # return count

        # 在遇到固定位置的判断时，尝试多多考虑二进制来对位置进行保存
        # 官方思路：使用二进制来保存状态，并且通过题意，1,10两个位置的存在与否都没有影响
        # 放置一共只有3中形式，2,3,4,5 4,5,6,7 5,6,7,8 这三种
        from collections import defaultdict
        left, middle, right = 0b11110000, 0b11000011, 0b00001111
        occupied = defaultdict(int)
        for seat in reservedSeats:
            # 只需要保存出现在2,9之间的被占位置，转化为二进制的对应位置
            if 2 <= seat[1] <= 9:
                # 此处或的作用相当于加，将之前的1位置添加到当前的数中
                occupied[seat[0]] |= (1 << (seat[1] - 2))

        # 剩余行的直接增2
        ans = (n - len(occupied)) * 2
        for row, bitmask in occupied.items():
            # 分别对三种状态使用或来进行判断
            if (bitmask | left) == left or (bitmask | middle) == middle or (bitmask | right) == right:
                ans += 1
        return ans

    def numTeams(self, rating: List[int]) -> int:
        # 官方还有一种思路，即以中间位置的值作为基准，左边查找比其小/大的值，右边也做同样操作
        # 这样可以得到以其为基准时所满足的数量为 cl*cr + (l-cl)*(r-cr) 这样也不用额外的空间复杂度

        # 设置两个列表分别保存后面所出现的元素比当前值大/小的数量
        # 将时间复杂度降为O(n2)
        lens = len(rating)
        if lens < 3:
            return 0
        lower, bigger = [0] * lens, [0] * lens
        for i in range(lens - 1):
            for j in range(i + 1, lens):
                lower[i] += 1 if rating[i] > rating[j] else 0
                bigger[i] += 1 if rating[i] < rating[j] else 0

        count = 0
        for i in range(lens - 2):
            for j in range(i + 1, lens - 1):
                # 在进行最终遍历时只需要自增之前的数量列表即可
                if rating[i] > rating[j]:
                    count += lower[j]
                elif rating[i] < rating[j]:
                    count += bigger[j]

        return count

    def processQueries(self, queries: List[int], m: int) -> List[int]:
        # 直接暴力不BB，模拟操作
        # 时间复杂度为O(nm)
        arr = [i for i in range(1, m + 1)]
        res = []
        for query in queries:
            pos = arr.index(query)
            res.append(pos)
            arr.pop(pos)
            arr.insert(0, query)

        return res

    def findMinFibonacciNumbers(self, k: int) -> int:
        # 先构建fab数列，再用贪心思想的二分查找寻找当前小于等于k的值，并将k减去相应的值
        # 时间复杂度为O(n)+O(logn)
        lens = 2
        fab = [1, 1]
        while fab[lens - 1] <= k:
            fab.append(fab[lens - 1] + fab[lens - 2])
            lens += 1

        k -= fab[lens - 2]
        count = 1
        left, right = 0, lens - 1
        while k:
            while left < right:
                mid = left + (right - left) // 2
                if k >= fab[mid]:
                    left = mid + 1
                else:
                    right = mid

            k -= fab[left - 1]

            count += 1
            right, left = left, 0

        return count

    def maxScore(self, cardPoints: List[int], k: int) -> int:
        # 这DP写得有问题
        # lens = len(cardPoints)
        # max_val, time = 0, 0
        # dp = [[0] * (lens + 1) for _ in range(lens+1)]
        # for i in range(lens):
        #     for j in range(lens):
        #         dp[i+1][j+1] = max(dp[i][j+1] , dp[i+1][j]) + max(cardPoints[i], cardPoints[lens - j-1])
        #         if i + j+1 == k:
        #             max_val = max(max_val, dp[i+1][j+1])
        #             time += 1
        #             break
        #
        #         if time == 2:
        #             return max_val
        # return max_val

        # 官方思路：前缀和+滑动窗口+逆向思维
        # 既然求取的是两边的最大，则可以转化为求取中间的最小，再减去即可
        lens = len(cardPoints)
        pre_sum = [0] * (lens + 1)
        for i in range(lens):
            pre_sum[i + 1] = pre_sum[i] + cardPoints[i]

        # 求取中间的最小
        min_sum = pre_sum[lens]
        for i in range(lens - k, lens + 1):
            min_sum = min(min_sum, pre_sum[i] - pre_sum[i - lens + k])

        # 最后减去即可
        return pre_sum[lens] - min_sum

    def findDiagonalOrder(self, nums: List[List[int]]) -> List[int]:
        # 使用行列和进行分类，然后逆序即可
        # 时间复杂度为O(nm)，空间复杂度为O(nm)
        row = len(nums)
        col = len(nums[0])
        for num in nums[1:]:
            col = max(col, len(num))

        # 此处有个小坑，最后行，列的值不一定是总和最大的
        store = [[] for _ in range(row + col - 1)]
        for i, num in enumerate(nums):
            for j, dig in enumerate(num):
                store[i + j].append(dig)

        res = []
        for l in store:
            res += l[::-1]

        return res

    def kLengthApart(self, nums: List[int], k: int) -> bool:
        # 对于特殊情况的判断
        if not k:
            return True
        # 思想是记录前一个1的位置，然后他们之间的0数量就能通过下标直接求解
        pre_index = -k - 1
        for i, value in enumerate(nums):
            if value:
                # 记录数量少的时候直接返回False即可
                if i - pre_index - 1 < k:
                    return False
                pre_index = i
        return True

    def longestSubarray(self, nums: List[int], limit: int) -> int:
        # 这个方法以前确实没见过，慢慢学（使用单调队列可以维护一个动态的最大最小值）
        # 维护两个单调队列，一个保存降序元素，一个保存升序元素，两个都包含下标
        # 使用滑动窗口，如果当前max-min大于limit，则需要将开始位置后移
        # 移动时候，如果出现队首元素比开始位置靠前时，将队首元素出列
        up_sort, down_sort = [], []
        # 单调队列当前的队首位置
        up_p, down_p = 0, 0
        # 滑动窗口的起点与最终结果的取值
        st, res = 0, 0
        for i, num in enumerate(nums):
            # 当len与队首相等时，表示已经超出了范围
            while len(up_sort) > up_p and num < up_sort[-1][0]:
                up_sort.pop()
            up_sort.append([num, i])

            while len(down_sort) > down_p and num > down_sort[-1][0]:
                down_sort.pop()
            down_sort.append([num, i])

            # 在出现子数组已经不满足当前的limit时，需要开始移动左指针
            while down_sort[down_p][0] - up_sort[up_p][0] > limit:
                st += 1
                # 发现当前的队列中的元素已经在范围之外时，则将首部指针右移
                if up_sort[up_p][1] < st:
                    up_p += 1
                if down_sort[down_p][1] < st:
                    down_p += 1

            res = max(res, i - st + 1)

        return res


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
    # print(show.subarraySum([4,5,0,-2,-3,1],5))

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

    # 769 最多能完成排序的块
    # print(show.maxChunksToSorted([3, 2, 4, 0, 1, 6, 5]))

    # 775 全局倒置与局部倒置
    # print(show.isIdealPermutation([6,4,2,1,0,3,5]))

    # 792 匹配子序列的单词数
    # print(show.numMatchingSubseq('abcde', ["a", "bb", "acd", "ace", 'def']))

    # 795 区间子数组个数
    # print(show.numSubarrayBoundedMax([73, 55, 36, 5, 55, 14, 9, 7, 72, 52], 32, 69))

    # 825 适龄的朋友
    # print(show.numFriendRequests([20, 30, 100, 110, 120]))

    # 835 图像重叠
    # print(show.largestOverlap([[1, 1, 0], [0, 1, 0], [0, 1, 0]], [[0, 0, 0], [0, 1, 1], [0, 0, 1]]))

    # 870 优势洗牌
    # print(show.advantageCount([2, 0, 4, 1, 2], [1, 3, 0, 0, 2]))

    # 873 最长的斐波那契子序列的长度
    # print(show.lenLongestFibSubseq([2, 4, 7, 8, 9, 10, 14, 15, 18, 23, 32, 50]))

    # 900 RLE迭代器
    # obj = RLEIterator([3, 8, 0, 9, 2, 5])
    # print(obj.next(2))
    # print(obj.next(1))
    # print(obj.next(1))

    # 907 子数组的最小值之和，难度等级拉满（单调栈的应用）
    # print(show.sumSubarrayMins([85, 93, 93, 90]))

    # 915 分割数组
    # print(show.partitionDisjoint([90, 47, 69, 10, 43, 92, 31, 73, 61, 97]))

    # 918 环形子数组的最大和（难度值拉满，主要还是见识太少）
    # print(show.maxSubarraySumCircular([-5, -1, -4]))

    # 926 将字符串翻转到单调递增
    # print(show.minFlipsMonoIncr("10011111110010111011"))

    # 945 使数组唯一的最小增量
    # print(show.minIncrementForUnique([0, 2, 2]))

    # 950 按递增顺序显示卡牌
    # print(show.deckRevealedIncreasing([17, 13, 11, 2, 3, 5, 7]))

    # 954 二倍数对数组
    # print(show.canReorderDoubled([-4, -1, 1, 2, 2, 3, 4, 6]))

    # 962 最大宽坡度
    # print(show.maxWidthRamp([6, 0, 8, 2, 1, 5]))

    # 969 煎饼排序
    # print(show.pancakeSort([1, 2, 3]))

    # 974 和可被K整除的子数组
    # print(show.subarraysDivByK([4, 5, 0, -2, -3, 1], 5))

    # 978 最长湍流子数组
    # print(show.maxTurbulenceSize([4, 8]))

    # 1007 行相等的最少多米诺旋转
    # print(show.minDominoRotations([2, 1, 2, 4, 2, 2], [5, 2, 6, 2, 3, 2]))

    # 1011 在D天内送达包裹的能力
    # print(show.shipWithinDays([1, 2, 3, 1, 1], 4))

    # 1014 最佳观光组合
    # print(show.maxScoreSightseeingPair([8, 3, 4]))

    # 1031 两个非重叠子数组的最大和
    # print(show.maxSumTwoNoOverlap([2, 1, 5, 6, 0, 9, 5, 0, 3, 8], 3, 4))

    # 1035 不相交的线
    # print(show.maxUncrossedLines([1, 4, 2], [1, 2, 4]))

    # 1040 移动石子直到连续II （个人感觉极难，思路根本想不完全）
    # print(show.numMovesStonesII([6, 7, 8, 1]))

    # 1052 爱生气的书店老板
    # print(show.maxSatisfied([1, 0, 1, 2, 1, 1, 7, 5], [0, 1, 0, 1, 0, 1, 0, 1], 3))

    # 1053 交换一次的先前排列
    # print(show.prevPermOpt1([3, 1, 1, 4, 6]))

    # 1109 航班预订统计
    # print(show.corpFlightBookings([[1, 2, 10], [2, 3, 20], [2, 5, 25]], 5))

    # 1144 递减元素使数组呈锯齿状
    # print(show.movesToMakeZigzag([10, 4, 4, 10, 10, 6, 2, 3]))

    # 1146 快照数组
    # s = SnapshotArray(1)
    # s.set(0,16)
    # print(s.snap())
    # print(s.snap())
    # print(s.snap())
    # print(s.get(0,2))

    # 1169 查询无效交易
    # print(show.invalidTransactions(["alex,741,1507,barcelona","xnova,683,1149,amsterdam","bob,52,1152,beijing","bob,137,1261,beijing","bob,607,14,amsterdam","bob,307,645,barcelona","bob,220,105,beijing","xnova,914,715,beijing","alex,279,632,beijing"]))

    # 1177 构建回文串检测
    # print(show.canMakePaliQueries('abcda', [[3, 3, 0], [1, 2, 0], [0, 3, 1], [0, 3, 2], [0, 4, 1]]))

    # 1202 交换字符串中的元素
    # print(show.smallestStringWithSwaps('dcabe', [[0, 3], [0, 1], [2, 4]]))

    # 1208 尽可能使字符串相等
    # print(show.equalSubstring('abcd', 'cdef', 3))

    # 1222 可以攻击国王的皇后 print(show.queensAttacktheKing( [[5, 6], [7, 7], [2, 1], [0, 7], [1, 6], [5, 1], [3, 7], [0, 3],
    # [4, 0], [1, 2], [6, 3], [5, 0], [0, 4], [2, 2], [1, 1], [6, 4], [5, 4], [0, 0], [2, 6], [4, 5], [5, 2], [1, 4],
    # [7, 5], [2, 3], [0, 5], [4, 2], [1, 0], [2, 7], [0, 1], [4, 6], [6, 1], [0, 6], [4, 3], [1, 7]], [3, 4]))

    # 1233 删除子文件夹
    # print(show.removeSubfolders(["/a", "/a/b", "/c/d", "/c/d/e", "/c/f", "/c/de"]))

    # 1267 统计参与通信的服务器
    # 这个题值得思考，尤其是并查集的使用部分
    # print(show.countServers([[0], [0], [1]]))

    # 1277 统计全为1的正方形子矩阵
    # print(show.countSquares([[0, 1, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1], [1, 0, 1, 0]]))

    # 1292 元素和小于等于阈值的正方形的最大边长 print(show.maxSideLength([[2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2,
    # 2, 2], [2, 2, 2, 2, 2]], 1))

    # 1296 划分数组为连续数字的集合
    # print(show.isPossibleDivide([1, 1, 2, 2, 3, 3], 2))

    # 1300 转变数组后最接近目标值的数组和
    # print(show.findBestValue([2, 3, 5], 10))

    # 1329 将矩阵按对角线排序
    # print(show.diagonalSort([[3, 3, 1, 1], [2, 2, 1, 2], [1, 1, 1, 2]]))

    # 1333 餐厅过滤器
    # print(show.filterRestaurants(
    #     [[77484, 13400, 1, 4010, 2926], [3336, 85138, 0, 49966, 89979], [28391, 55328, 0, 69158, 29058],
    #      [57395, 64988, 0, 45312, 30261]], 0, 99739, 60242))

    # 1338 数组大小减半
    # print(show.minSetSize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

    # 1343 大小为k且平均值大于等于阈值的子数组数目
    # print(show.numOfSubarrays([11, 13, 17, 23, 29, 31, 7, 5, 2, 3], 3, 5))

    # 1366 通过投票对团队排名 print(show.rankTeams( ["FVSHJIEMNGYPTQOURLWCZKAX", "AITFQORCEHPVJMXGKSLNZWUY",
    # "OTERVXFZUMHNIYSCQAWGPKJL", "VMSERIJYLZNWCPQTOKFUHAXG", "VNHOZWKQCEFYPSGLAMXJIUTR", "ANPHQIJMXCWOSKTYGULFVERZ",
    # "RFYUXJEWCKQOMGATHZVILNSP", "SCPYUMQJTVEXKRNLIOWGHAFZ", "VIKTSJCEYQGLOMPZWAHFXURN", "SVJICLXKHQZTFWNPYRGMEUAO",
    # "JRCTHYKIGSXPOZLUQAVNEWFM", "NGMSWJITREHFZVQCUKXYAPOL", "WUXJOQKGNSYLHEZAFIPMRCVT", "PKYQIOLXFCRGHZNAMJVUTWES",
    # "FERSGNMJVZXWAYLIKCPUQHTO", "HPLRIUQMTSGYJVAXWNOCZEKF", "JUVWPTEGCOFYSKXNRMHQALIZ", "MWPIAZCNSLEYRTHFKQXUOVGJ",
    # "EZXLUNFVCMORSIWKTYHJAQPG", "HRQNLTKJFIEGMCSXAZPYOVUW", "LOHXVYGWRIJMCPSQENUAKTZF", "XKUTWPRGHOAQFLVYMJSNEIZC",
    # "WTCRQMVKPHOSLGAXZUEFYNJI"]))

    # 1375 灯泡开关III
    # print(show.numTimesAllBlue([2, 1, 4, 3, 6, 5]))

    # 1386 安排电影院座位
    # print(show.maxNumberOfFamilies(4, [[2, 10], [3, 1], [1, 2], [2, 2], [3, 5], [4, 1], [4, 9], [2, 7]]))

    # 1395 统计作战单位数
    # print(show.numTeams([1, 2, 3, 4]))

    # 1409 查询带键的排列
    # print(show.processQueries([7, 5, 5, 8, 3], 8))

    # 1414 和为k的最少斐波那契数字数目
    # print(show.findMinFibonacciNumbers(13))

    # 1423 可获得的最大点数
    # print(show.maxScore([100, 1, 1, 200, 1, 1], 3))

    # 1424 对角线遍历II
    # print(show.findDiagonalOrder([[14, 12, 19, 16, 9], [13, 14, 15, 8, 11], [11, 13, 1]]))

    # 1437 是否所有1至少相隔k个元素
    # print(show.kLengthApart([0, 1, 1, 1, 1, 1], 0))

    # 1438 绝对差不超过限制的最长连续子数组(有点难，是个新方法)
    # print(show.longestSubarray([10, 1, 2, 4, 7, 2], 5))
