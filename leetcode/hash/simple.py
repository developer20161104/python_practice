from typing import List


class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        # 空间复杂度为O(n)，时间复杂度为O(n)
        """
        p = set()
        for dig in nums:
            if dig in p:
                p.remove(dig)
            else:
                p.add(dig)

        return p.pop()
        """

        # 两个数之间的关系——异或
        # 时间复杂度O(n)，空间复杂度为O(1)
        """
        from functools import reduce
        return reduce(lambda x, y: x ^ y, nums)
        """
        # 异或大法好
        # 不调用库函数直接逐个异或
        cur = nums[0]
        for i in nums[1:]:
            cur ^= i
        return cur

    def isIsomorphic(self, s: str, t: str) -> bool:
        # 通过将字符映射到字典上，但是过于消耗内存
        # 并且由于对排序算法的使用，时间复杂度为O(n*log n)
        """
        from collections import defaultdict
        dict_1 = defaultdict(list)
        dict_2 = defaultdict(list)
        for i in range(len(s)):
            dict_1[s[i]].append(i)
            dict_2[t[i]].append(i)

        return sorted(dict_1.values()) == sorted(dict_2.values())
        """
        # 其实只需要将其映射到数字排列即可
        # 时间复杂度为O(n)
        def get_centralstr(st: str) -> str:
            # 构建映射关系字典
            maps = [0]*120
            ans = ""
            count = 1
            for ch in st:
                pos = ord(ch)-ord('a')
                # 对未存在于其中的构建映射
                if not maps[pos]:
                    maps[pos] = count
                    count += 1
                # 逐一添加关系
                ans += str(maps[pos])

            return ans

        # 由于python内建函数耗时太可怕，时间惨不忍睹
        return get_centralstr(s) == get_centralstr(t)
        # 相比之下直接映射似乎时间更短


if __name__ == '__main__':
    show = Solution()

    # 136 只出现一次的数字
    # print(show.singleNumber([4, 1, 2, 1, 2]))

    # 205 同构字符串
    # print(show.isIsomorphic("13", "42"))
