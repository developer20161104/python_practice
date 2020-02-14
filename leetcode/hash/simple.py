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
            maps = [0] * 126
            ans = ""
            count = 1
            for ch in st:
                pos = ord(ch)
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

    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False

        # 受制于hash表的长度
        maps = [0] * 126
        for ch in s:
            maps[ord(ch)] += 1

        # 只要出现不存在的元素即可返回判断
        for ch in t:
            if not maps[ord(ch)]:
                return False
            maps[ord(ch)] -= 1

        return True

    def wordPattern(self, pattern: str, strs: str) -> bool:
        # 关键在于建立两者之间的映射
        arr_list = strs.split()
        if len(pattern) != len(arr_list):
            return False

        # 此函数可以使用index内置函数来替换（妙）
        # 实现的方法为使用数字来作为中间映射，而次内置方法使用的是第一次出现的下标
        # 效果一致
        # list(map(pattern.index, pattern))
        def judge_str(arr: List[str]) -> str:
            count = 0

            # 注意到映射的字符串每个都得一一打印
            res = ""
            # 存储映射
            q = {}
            for ch in arr:
                if ch not in q:
                    q[ch] = str(count)
                    count += 1
                res += q[ch]

            return res

        # 相互映射必须两边都要判断
        return judge_str(list(pattern)) == judge_str(arr_list)

    def getHint(self, secret: str, guess: str) -> str:
        pos = [0] * 10
        count = [0] * 2
        res = []
        # 同时进行的话会有缺
        """
        for i in range(len(secret)):
            pos[int(secret[i])] += 1

            if pos[int(guess[i])]:
                if secret[i] == guess[i]:
                    count[0] += 1
                else:
                    count[1] += 1
                pos[int(guess[i])] -= 1

        return str(count[0])+'A'+str(count[1])+'B'
        """
        lens = len(secret)

        # 隐含的优先级关系，先找公牛再重新找母牛
        for i in range(lens):
            if secret[i] == guess[i]:
                count[0] += 1
            else:
                # 将不同位置的保留
                pos[int(secret[i])] += 1
                res.append(i)

        for i in res:
            # 查找母牛
            if pos[int(guess[i])]:
                count[1] += 1
                pos[int(guess[i])] -= 1

        return str(count[0]) + 'A' + str(count[1]) + 'B'

    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # set集合处理
        return list(set(nums1) & set(nums2))

    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        from collections import defaultdict

        # 通过字典映射来计数，然后再逐一查询求取
        dicts = defaultdict(int)
        ans = []
        # 先将值映射字典
        for i in nums1:
            dicts[i] += 1
        for i in nums2:
            # 再比对即可
            if i in dicts and dicts[i] > 0:
                dicts[i] -= 1
                ans.append(i)

        return ans

    def findTheDifference(self, s: str, t: str) -> str:
        from collections import defaultdict
        dicts = defaultdict(int)
        for e in s:
            dicts[e] += 1

        # 同样设置一个字典来查询即可
        for e in t:
            if e not in dicts or dicts[e] < 1:
                return e
            dicts[e] -= 1

        return ''

    def longestPalindrome(self, s: str) -> int:
        from collections import defaultdict
        dicts = defaultdict(int)
        for ch in s:
            dicts[ch] += 1

        # 注意到当为奇数时，也需要加上其中的最大子偶数
        ans, flag = 0, False
        for value in dicts.values():
            if not value % 2:
                ans += value
            else:
                # 可包含一个奇数作为中间位
                flag = True
                # 增加当前数中的最大偶数
                ans += value - 1

        return ans + 1 if flag else ans

    def numberOfBoomerangs(self, points: List[List[int]]) -> int:
        # 自动构建字典
        from collections import Counter

        tot = 0
        for p1 in points:
            # 统计点与点之间的距离，注意必须要计算所有点！（错误之处）
            count = Counter((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 for p2 in points)

            # 使用组合来进行统计
            for value in count.values():
                tot += value * (value - 1) // 2

        return tot * 2

    def islandPerimeter(self, grid: List[List[int]]) -> int:
        move = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        all = 0
        len_row, len_col = len(grid), len(grid[0])

        # 方法一
        # # 暴力查找四周来进行判断，时间感人
        # for i in range(len_row):
        #     for j in range(len_col):
        #
        #         for pos in move:
        #             cur = [i+pos[0], j+pos[1]]
        #             if grid[i][j] and not (0 <= cur[0] < len_row and 0 <= cur[1] < len_col and grid[cur[0]][cur[1]]):
        #                 all += 1
        #
        # return all

        # 考虑数学思想
        # c统计方块数，l统计下方与右方边界情况
        c, l = 0, 0
        for i in range(len_row):
            for j in range(len_col):
                # 只需要对岛屿部分进行判断
                if grid[i][j]:
                    c += 1
                    if i + 1 < len_row and grid[i + 1][j]:
                        l += 1
                    if j + 1 < len_col and grid[i][j + 1]:
                        l += 1

        # 利用对称性？
        return 4 * c - 2 * l


if __name__ == '__main__':
    show = Solution()

    # 136 只出现一次的数字
    # print(show.singleNumber([4, 1, 2, 1, 2]))

    # 205 同构字符串
    # print(show.isIsomorphic("13", "42"))

    # 242 有效的字母异位词
    # print(show.isAnagram("rat", "car"))

    # 290 单词规律
    # print(show.wordPattern("aba", "dog cat cat"))

    # 299 猜数字游戏
    # print(show.getHint("1123", "0111"))

    # 349两个数组的交集
    # print(show.intersection([1,2,2,1], [2,2]))

    # 350 两个数组的交集II
    # print(show.intersect([1,2], [1,1]))

    # 389 找不同
    # print(show.findTheDifference("abcd", "cdabw"))

    # 409 最长回文串
    # print(show.longestPalindrome('ccceeeeeabbadd'))

    # 447 回旋镖的数量
    # print(show.numberOfBoomerangs([[0,0],[1,0],[-1,0],[0,1],[0,-1]]))

    # 463 岛屿的周长
    # print(show.islandPerimeter([[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]))
