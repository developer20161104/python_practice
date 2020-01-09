from typing import  List

class Solution:
    # brute force
    # exceeding time
    '''
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i in range(len(nums)-1):
            for j in range(i+1,len(nums)):
                if nums[i] + nums[j] == target:
                    return [i, j]
        return None

    # error:no sort sequence

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        lens = len(nums)
        i, j = 0, lens-1
        while i < j:
            if nums[i]+nums[j] > target:
                j -= 1
            elif nums[i]+nums[j] < target:
                i += 1
            else:
                return [i, j]
    '''
    # use hash map to execute secondary circle(from n to 1)
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        maps = {}
        lens = len(nums)

        for i in range(lens):
            dif = target - nums[i]
            if dif in maps:
                return [maps[dif], i]
            # temporary save the front element
            maps[nums[i]] = i

        return []

if __name__ == '__main__':
    hello = Solution()
    print(hello.twoSum([2, 7, 11, 15], 9))