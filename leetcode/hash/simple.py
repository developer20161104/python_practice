from typing import List


# 705 设计 hash 集合
# 采用的是链表法解决hash冲突
class Node:
    def __init__(self, key):
        self.key = key
        self.next = None


class MyHashSet:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.lens = 1000
        # 设置头结点
        self.arr = [Node(-1) for _ in range(self.lens)]

    def add(self, key: int) -> None:
        pos = head = self.arr[key % self.lens]

        while pos.next is not None and pos.next.key != key:
            pos = pos.next

        # 用于判断是否要进行添加
        if pos.next is None:
            pos.next = Node(key)

        # 注意在结尾需要将修改好的链表重新链接到头结点上
        self.arr[key % self.lens] = head

    def remove(self, key: int) -> None:
        head = pos = self.arr[key % self.lens]
        while pos.next is not None and pos.next.key != key:
            pos = pos.next

        # 判断是否含有需要删除的key
        if pos.next is not None:
            pos.next = pos.next.next
        self.arr[key % self.lens] = head

    def contains(self, key: int) -> bool:
        """
        Returns true if this set contains the specified element
        """
        pos = self.arr[key % self.lens]
        while pos is not None and pos.key != key:
            pos = pos.next

        # 这句有点意思，判断言简意赅又没有漏掉各种条件
        return pos is not None


# except 版本（更加 pythonic）
# class Node:
#     def __init__(self, key):
#         self.key = key
#         self.next = None
#
#
# class MyHashSet:
#
#     def __init__(self):
#         """
#         Initialize your data structure here.
#         """
#         self.lens = 1000
#         self.arr = [Node(-1) for _ in range(self.lens)]
#
#     def add(self, key: int) -> None:
#         pos = head = self.arr[key % self.lens]
#
#         try:
#             while pos.next.key != key:
#                 pos = pos.next
#             # 当出现错误时pos指向的是倒数第二个值
#         except:
#             pos.next = Node(key)
#             self.arr[key % self.lens] = head
#
#     def remove(self, key: int) -> None:
#         head = pos = self.arr[key % self.lens]
#         try:
#             while pos.next.key != key:
#                 pos = pos.next
#             pos.next = pos.next.next
#             # 移除时出现错误则说明当前无此值
#         except:
#             self.arr[key % self.lens] = head
#
#     def contains(self, key: int) -> bool:
#         """
#         Returns true if this set contains the specified element
#         """
#         pos = self.arr[key % self.lens]
#         while pos is not None and pos.key != key:
#             pos = pos.next
#
#         return pos is not None


# 706 设计hash映射
# 同样可以采用链地址法解决冲突
class Pair:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = None


class MyHashMap:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.lens = 2000
        # 设置头链表
        self.arr = [Pair(-1, -1) for _ in range(self.lens)]

    def put(self, key: int, value: int) -> None:
        """
        value will always be non-negative.
        """
        head = cur_pos = self.arr[key % self.lens]
        try:
            while cur_pos.next.key != key:
                cur_pos = cur_pos.next
            cur_pos.next.value = value
        except AttributeError:
            cur_pos.next = Pair(key, value)
            self.arr[key % self.lens] = head

    def get(self, key: int) -> int:
        """
        Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
        """
        cur_pos = self.arr[key % self.lens]
        try:
            while cur_pos.next.key != key:
                cur_pos = cur_pos.next
            return cur_pos.next.value
        # 报错原因是找到尾部None值
        except AttributeError:
            return -1

    def remove(self, key: int) -> None:
        """
        Removes the mapping of the specified value key if this map contains a mapping for the key
        """
        head = cur_pos = self.arr[key % self.lens]
        try:
            while cur_pos.next.key != key:
                cur_pos = cur_pos.next
            cur_pos.next = cur_pos.next.next
        except AttributeError:
            self.arr[key % self.lens] = head


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

    def findWords(self, words: List[str]) -> List[str]:
        # 采用的是字典的映射
        arr = ['qwertyuiop', 'asdfghjkl', 'zxcvbnm']
        dicts = {}
        res = []

        # 逐一更新字典
        for i in range(3):
            dicts.update(dict.fromkeys(arr[i], i))

        for strs in words:
            lens = len(strs)
            # 判断是否为空
            if not lens:
                continue
            pre = dicts[strs[0].lower()]

            i = 0
            while i < lens:
                # 只要出现一个不同的即排除
                if dicts[strs[i].lower()] != pre:
                    break
                pre = dicts[strs[i].lower()]
                i += 1

            if i == lens:
                res.append(strs)

        return res

    def distributeCandies(self, candies: List[int]) -> int:
        # 没必要用dict来存储，用set即可（妙啊）
        # from collections import Counter
        # tot = Counter(candies).values()
        #
        # return min(len(candies)//2, len(tot))
        return min(len(candies) // 2, len(set(candies)))

    def findLHS(self, nums: List[int]) -> int:
        # 方法一：字典映射排序法 O(n*log n)
        # from collections import Counter
        # # 先找出键值对映射关系
        # d = sorted(Counter(nums).items())
        #
        # cur_max, lens = 0, len(d)
        # if lens < 2:
        #     return cur_max
        #
        # # 再逐一比对，慢出天际
        # for i in range(1, lens):
        #     if abs(d[i][0] - d[i-1][0]) <= 1:
        #         cur_max = max(cur_max, d[i][1] + d[i-1][1])
        #
        # return cur_max

        # 方法二
        # 只需要一次遍历即可，但是速度还是感人？？？
        # from collections import defaultdict
        # dicts = defaultdict(int)
        #
        # lens = len(nums)
        # res = 0
        # for i in range(lens):
        #     此处小技巧，直接对结果进行查询即可
        #     dicts[nums[i]] += 1
        #     if dicts[nums[i]+1] != 0:
        #         res = max(res, dicts[nums[i]+1]+dicts[nums[i]])
        #     if dicts[nums[i]-1] != 0:
        #         res = max(res, dicts[nums[i]-1]+dicts[nums[i]])
        #
        # return res

        # 方法三
        from collections import Counter
        # 构建字典
        d = Counter(nums)
        res = 0
        for item in d:
            # 只需要查找差为1的是否在字典即可，O(n)
            if item + 1 in d:
                res = max(res, d[item] + d[item + 1])

        return res

    def findRestaurant(self, list1: List[str], list2: List[str]) -> List[str]:
        d = {}
        for pos, add in enumerate(list1):
            d[add] = pos

        # 方法一：使用内库建立defaultdict存储后再查找min
        # from collections import defaultdict
        # save = defaultdict(list)
        # for pos, ch in enumerate(list2):
        #     if ch in d:
        #         save[d[ch]+pos].append(ch)
        #
        # return save[min(save.keys())]

        # 方法二，直接查找
        cur_min = 10000
        res = []
        for pos, ch in enumerate(list2):
            if ch in d:
                tmp = d[ch] + pos
                if tmp < cur_min:
                    res.clear()
                    res.append(ch)
                    cur_min = tmp
                elif tmp == cur_min:
                    res.append(ch)

        return res

    def getImportance(self, employees, id: int) -> int:
        from collections import defaultdict, deque
        d = defaultdict(list)
        q = deque()

        # 先构建hash映射
        for e in employees:
            d[e[0]].append(e[1])
            d[e[0]].append(e[2])

        # 非递归的广度优先遍历（BFS）
        q.append(d[id][1])
        tot = d[id][0]
        while q:
            cur_list = q.popleft()
            # 依次添加员工重要性以及下属id即可
            for cur_id in cur_list:
                q.append(d[cur_id][1])
                tot += d[cur_id][0]

        return tot

    def longestWord(self, words: List[str]) -> str:
        # 方法一：暴力搜索
        # # 暴力法查找
        # dicts = set(words)
        # cur_max = 0
        # res = []
        #
        # for cur_str in words:
        #     lens, i = len(cur_str), 0
        #     # 逐一比对
        #     while i < lens-1:
        #         if cur_str[:i+1] not in dicts:
        #             break
        #         i += 1
        #
        #     # 截取最长字符串
        #     if i == lens-1:
        #         if lens > cur_max:
        #             res.clear()
        #             res.append(cur_str)
        #             cur_max = lens
        #         elif lens == cur_max:
        #             res.append(cur_str)
        #
        # # 找出字典序最小
        # return min(res)

        # 方法二：trie树（前缀树）
        from collections import defaultdict
        from functools import reduce

        # 构建迭代器
        Trie = lambda: defaultdict(Trie)
        trie = Trie()
        END = True

        # 神仙操作
        for i, word in enumerate(words):
            # 使用reduce对每个 word 中的 ch 建立字典，作为命名
            # reduce实现迭代，实现字典中的嵌套字典，并在结尾字典设置key=END，value=i作为终止符
            # 每个word单独嵌套，其顶层皆为trie的一个键，值依次迭代
            reduce(dict.__getitem__, word, trie)[END] = i
            # print(trie)

        stack = list(trie.values())
        ans = ""
        # 由于嵌套本身已经有序，因此找出最长word即可
        while stack:
            cur = stack.pop()
            if END in cur:
                word = words[cur[END]]
                if len(word) > len(ans) or len(word) == len(ans) and word < ans:
                    ans = word
                stack.extend([cur[letter] for letter in cur if letter != END])

        return ans

    def shortestCompletingWord(self, licensePlate: str, words: List[str]) -> str:
        from collections import defaultdict
        d = defaultdict(int)
        count = 0
        # 先建立字典映射
        for ch in licensePlate:
            if ch.isalpha():
                d[ch.lower()] += 1
                count += 1

        cur_len = 10000
        cur_word = ""
        # 再逐一单词进行匹配
        for word in words:
            check = d.copy()
            tot_len = count

            lens = len(word)
            for i in range(lens):
                if check[word[i]] > 0:
                    tot_len -= 1
                    check[word[i]] -= 1

            # 保留能够完全匹配的单词
            if tot_len == 0 and cur_len > lens:
                cur_len = lens
                cur_word = word

        return cur_word

    def numJewelsInStones(self, J: str, S: str) -> int:
        # 速度感人
        # s = set(J)
        # return len([x for x in S if x in s])
        s = set(J)
        count = 0
        # 在set中逐一查找即可
        for cur in S:
            if cur in s:
                count += 1

        return count

    def subdomainVisits(self, cpdomains: List[str]) -> List[str]:
        from collections import defaultdict
        dicts = defaultdict(int)

        for ele in cpdomains:
            # 先切分为计数值与网址
            tot = ele.split()
            # 再将网址细分
            cur_net = tot[1].split('.')
            count = int(tot[0])
            # 共有部分
            dicts[tot[1]] += count
            dicts[cur_net[-1]] += count
            # 由于只有两种网址，采用条件判断即可
            if len(cur_net) == 3:
                dicts[cur_net[1] + '.' + cur_net[2]] += count

        return [str(value) + " " + key for key, value in dicts.items()]

    def uncommonFromSentences(self, A: str, B: str) -> List[str]:
        from collections import Counter
        # 建立两个counter相互比较
        # count_A, count_B = Counter(A.split()), Counter(B.split())
        # return [x for x in count_A if count_A[x] == 1 and count_B[x] == 0] + \
        #        [x for x in count_B if count_B[x] == 1 and count_A[x] == 0]

        # 为什么不把它们合并呢？？？
        # 只需要出现一次即可满足要求
        count = Counter(A.split() + B.split())
        return [x for x in count if count[x] == 1]

    def isAlienSorted(self, words: List[str], order: str) -> bool:
        len_order, len_word = len(order), len(words)
        dicts = {}

        # 先有序存储
        for i in range(len_order):
            dicts[order[i]] = i

        for i in range(1, len_word):
            pre_len = len(words[i - 1])

            # 再逐一比对
            try:
                for j in range(pre_len):
                    if dicts[words[i - 1][j]] > dicts[words[i][j]]:
                        return False
                    elif dicts[words[i - 1][j]] < dicts[words[i][j]]:
                        break
            # 当下标越界时，表示前面的字符数量比后面还多，此时必为错误
            except IndexError:
                return False

        return True

    def repeatedNTimes(self, A: List[int]) -> int:
        # error: [8323]
        # arr = A[:2]
        # for e in A:
        #     if e == arr[1]:
        #         arr[0] = e
        #     else:
        #         arr[1] = e
        #
        # return arr[0]

        # 数学方法比较耗时
        # lens = len(A)
        # return (sum(A)-sum(set(A)))//(lens//2-1)

        # 有点意思
        # 要求的元素出现了N次，代表其它的只出现一次，这是关键
        lens = len(A)
        for i in range(lens-2):
            # 由于有N个，代表此元素必在两个或者一个之间相邻
            if A[i] == A[i+1] or A[i] == A[i+2]:
                return A[i]

        # 如果不满足以上情况，则必在末端
        # example：8238
        return A[-1]


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

    # 500 键盘行
    # print(show.findWords(["Hello", "Alaska", "Dad", "Peace"]))

    # 575 分糖果
    # print(show.distributeCandies([1,1,2,2,2,3,3,3]))

    # 594 最长和谐子序列
    # print(show.findLHS([1,3,4,2]))

    # 599  两个列表的最小索引总和
    # print(show.findRestaurant(["Shogun", "KFC", "Burger King", "Tapioca Express"], ["KFC", "Shogun", "Burger King"]))

    # 690 员工重要性
    # print(show.getImportance([[1, 5, [2, 3]], [2, 3, []], [3, 3, []]], 1))

    # 720 词典中最长的单词
    # print(show.longestWord(["k","lg","it","oidd","oid","oiddm","kfk","y","mw","kf","l","o","mwaqz","oi","ych","m","mwa"]))

    # 748 最短完整词
    # print(show.shortestCompletingWord("1s3 456", ["looks", "pest", "stew", "show"]))

    # 771 宝石与石头
    # print(show.numJewelsInStones("aA", "aAAbbbb"))

    # 811 子域名访问计数
    # print(show.subdomainVisits(["900 google.mail.com", "50 yahoo.com", "1 intel.mail.com", "5 wiki.org"]))

    # 884 两句话中的不常见单词
    # print(show.uncommonFromSentences("this apple is sweet", "this apple is sour"))

    # 953 验证外星语词典
    # print(show.isAlienSorted(["word","world","row"], "worldabcefghijkmnpqstuvxyz"))

    # 961 重复N次的元素
    # print(show.repeatedNTimes([5, 1, 5, 2, 5, 3, 5, 4, 5, 2]))
