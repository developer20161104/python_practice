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
        left, right = 0, len(nums)-1

        # 由于需要进行压缩，所以必须判断左边与右边相等的情况
        while left <= right:
            mid = left + (right-left)//2

            if nums[mid] == target:
                return mid
            # 左边首尾是升序时
            elif nums[left] <= nums[mid]:
                # 当出现相等时不断压缩
                # 寻找较优判断条件，其余丢到else里面即可
                if nums[left] <= target <= nums[mid]:
                    right = mid-1
                else:
                    left = mid+1
            else:
                # 否则就是右边升序！（围点旋转）
                if nums[mid] <= target <= nums[right]:
                    left = mid+1
                else:
                    right = mid-1

        return -1


# 二分查找最优方法，保留左闭右开原则
def binary_search(arr: List[int], target: int) -> int:
    left, right = 0, len(arr)

    # 无需判断相等的情况，只需要在最后判别结果是否真正存在即可
    while left < right:
        mid = left + (right-left)//2
        if arr[mid] < target:
            left = mid+1
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
