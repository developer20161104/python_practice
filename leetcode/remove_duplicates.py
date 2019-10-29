from typing import List

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        lens = len(nums)
        if lens == 0:
            return 0
        # recording current element index
        st = 1
        # previous element
        pre = nums[0]
        # increment index
        i = 1
        while i < lens:
            # recording duplicate element
            k = 0
            while i+k < lens and nums[i+k] == pre:
                k += 1
            # avoid out ot index
            if i+k >= lens:
                break
            # filling element
            nums[st] = nums[i+k]
            st += 1
            pre = nums[i+k]
            i += k
        # print(nums)
        return st

    def removeElement(self, nums: List[int], val: int) -> int:
        # k为多余的统计步骤
        '''
        lens = len(nums)
        k, pos = 0, 0
        for i in range(lens):
            if nums[i] == val:
                k += 1
            else:
                nums[pos] = nums[i]
                pos += 1

        # print(nums)
        return lens-k
        '''
        # find the position and insert into it directly
        lens = len(nums)
        pos = 0
        for i in range(lens):
            if nums[i] != val:
                nums[pos] = nums[i]
                pos += 1
        return  pos

    def searchInsert(self, nums: List[int], target: int) -> int:
        # brute force
        '''
        lens = len(nums)
        for i in range(lens):
            if nums[i] >= target:
                return i
        return lens
        '''
        # <=
        lens = len(nums)
        left, right = 0, lens
        # binary search template
        # right length, not equal, remove which median
        while left < right:
            mid = left + (right - left)//2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid+1
            else:
                # keep right boundary
                right = mid

        # both left and right is fine
        # the final check is needed when the result is not exit
        return right

    def maxSubArray(self, nums: List[int]) -> int:
        # no max value
        '''
        lens = len(nums)
        if lens == 0:
            return 0
        ans = 0
        i, k = 0, 0
        while i < lens:
            ans += nums[i]
            if ans < 0:
                if k != 0:
                    ans = nums[i-k]
                    i -= k
                    k = 0
                else:
                    ans = nums[i]
            else:
                k += 1

            i += 1
        return ans
        '''

        # answer
        ans = nums[0]
        sums = 0
        for i in nums:
            # sums + i > i
            # positive effect
            if sums > 0:
                sums += i
            else:
                sums = i
            ans = max(ans, sums)

        return ans

    def plusOne(self, digits: List[int]) -> List[int]:
        lens = len(digits)
        # wrong
        '''
        for i in range(lens-1, 0, -1):
            if digits[i]+incre >= 9:
                digits[i] = 0
                incre += 1
            else:
                digits[i] = digits[i] + incre + 1
                incre = 0

        if digits[0]+incre == 9:
            digits[0] = 0
            digits.insert(0, 1)
        else:
            digits[0] = digits[0] + 1 + incre

        print(digits)
        return digits
        '''
        # increment directly
        if digits[lens-1] < 9:
            digits[lens-1] += 1
        else:
            # special circumstances
            incre = 1
            for i in range(lens-1, 0, -1):
                if digits[i]+incre > 9:
                    digits[i] = 0
                else:
                    digits[i] += incre
                    incre = 0

            # judge the first element
            if incre == 1 and digits[0] == 9:
                digits[0] = 0
                digits.insert(0, 1)
            elif incre == 1:
                digits[0] += 1

        # print(digits)
        return digits

    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        i, j = 0, 0
        lens = m+n
        # special circumstances
        if n == 0:
            return
        elif m == 0:
            for i in range(n):
                nums1[i] = nums2[i]
            return

        while i < m and j < n:
            # whether i is larger than j
            if nums1[i] >= nums2[j]:
                # exchange position
                for k in range(lens-1,i,-1):
                    nums1[k] = nums1[k-1]
                # insert value
                nums1[i] = nums2[j]
                j += 1
                m += 1

            i += 1

        # larger than all value
        if j < n and nums2[j] > nums1[m-1]:
            while j < n:
                nums1[i] = nums2[j]
                i += 1
                j += 1
        print(nums1)

    def generate(self, numRows: int) -> List[List[int]]:
        # public answer
        '''
        triangle = []

        for row_num in range(numRows):
            # The first and last row elements are always 1.
            # init list and padding elements
            row = [None for _ in range(row_num+1)]
            row[0], row[-1] = 1, 1

            # Each triangle element is equal to the sum of the elements
            # above-and-to-the-left and above-and-to-the-right.
            for j in range(1, len(row)-1):
                row[j] = triangle[row_num-1][j-1] + triangle[row_num-1][j]

            triangle.append(row)

        return triangle
        '''
        lists = []
        # 仅将第一位作为起始初始化，后续皆可循环推导
        if numRows >= 1:
            lists.append([1])
        for i in range(1, numRows):
            # 保存先前列表便于操作
            pre = lists[-1]
            cur_sort = [1]
            lens = len(pre)
            for j in range(1, lens):
                cur_sort.append(pre[j] + pre[j-1])
            cur_sort.append(1)
            lists.append(cur_sort)

        return lists

    def getRow(self, rowIndex: int) -> List[int]:
        lists = []
        if rowIndex >= 0:
            lists.append(1)

        for i in range(0, rowIndex):
            cur = [1]
            pre = lists
            lens = len(pre)
            for j in range(1, lens):
                cur.append(pre[j] + pre[j-1])
            cur.append(1)
            lists = cur
        return lists

if __name__ == '__main__':
    show = Solution()
    print(show.removeDuplicates([0,0,1,1,1,2,2,3,3,4]))

    print(show.removeElement([0,1,2,2,3,0,4,2], 2))

    print(show.searchInsert([1,3,5,6], 7))

    print(show.maxSubArray([-2,1,-3,4,-1,2,1,-5,4]))

    print(show.plusOne([9,9,9,9]))

    print(show.merge([0],0,[1],1))

    print(show.generate(5))

    print(show.getRow(3))