from typing import List


class Solution:
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
            if i > start and candidates[i-1] == candidates[i]:
                continue

            path.append(candidates[i])

            # 在进行递归的时候以下一个点为起始值
            self._dfs2(candidates, target - candidates[i], path, res, i+1)
            # 回溯模板
            path.pop()


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
