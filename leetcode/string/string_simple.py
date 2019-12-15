from typing import List


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def romanToInt(self, s: str) -> int:
        # 建立字典表进行映射判断
        dicts = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
        special = {"IV": 4, "IX": 9, "XL": 40, "XC": 90, "CD": 400, "CM": 900}

        count, lens, sp = 0, len(s), 0
        # 此类遍历会有空缺：如果特殊组合出现在奇数对时，会找不到
        """
        for sp in range(1, lens, 2):
            if s[sp - 1:sp + 1] in special:
                count += special[s[sp - 1:sp + 1]]
            else:
                count += dicts[s[sp - 1]] + dicts[s[sp]]

        return count+dicts[s[-1]] if lens % 2 else count
        """
        # 遍历查找
        while sp < lens:
            # 出现特殊字符
            if sp + 1 < lens and s[sp:sp + 2] in special:
                count += special[s[sp:sp + 2]]
                sp += 2
            else:
                # 正常情况
                count += dicts[s[sp]]
                sp += 1

        return count

    def longestCommonPrefix(self, strs: List[str]) -> str:
        # 暴力搜索法
        """
        count_str, lens = "", len(strs)
        if not lens:
            return count_str

        len_int = len(strs[0])

        if lens == 1:
            return strs[0]
        # 逐一字符进行比较
        for ch in range(len_int):
            count = 1
            for sot in strs[1:]:
                if ch >= len(sot):
                    break
                if strs[0][ch] == sot[ch]:
                    count += 1
            if count == lens:
                count_str += strs[0][ch]
            else:
                break

        return count_str
        """
        # 大神思路：字符串的排序比较
        # 比较最大最小公共前缀可以转化为列表中的最大最小（以ASCII码）两个字符串的比较
        if not strs:
            return ""
        front, end = min(strs), max(strs)
        for i, x in enumerate(front):
            if x != end[i]:
                return front[:i]
        return front

    def isValid(self, s: str) -> bool:
        # 使用栈来处理，并利用一个字典映射两边括号之间的关系即可
        dicts = {"(": ")", "[": "]", "{": "}"}

        if not len(s):
            return True
        stack = []
        for sym in s:
            if sym == "(" or sym == "{" or sym == "[":
                stack.append(sym)
            else:
                # 注意此处需要判断当前的栈是否为空，即s中只含有右括号时
                if not len(stack) or dicts[stack.pop()] != sym:
                    return False

        return False if len(stack) else True

    def strStr(self, haystack: str, needle: str) -> int:
        # 调用库函数解法
        return haystack.index(needle) if needle in haystack else -1

    def countAndSay(self, n: int) -> str:
        if n < 1:
            return ""

        # 使用set会将后面元素与之前的合并
        """
        pre_str = "1"
        for i in range(2, n+1):
            cur_set, cur_str = set(pre_str), ""
            while len(cur_set):
                cur_val = cur_set.pop()
                cur_str += str(pre_str.count(cur_val)) + cur_val

            pre_str = cur_str

        return pre_str
        """
        # 只能通过遍历进行判断
        pre_str = "1"
        for _ in range(2, n + 1):
            # all_val记录当前保存的输出
            all_val, cur_val, count_val = "", pre_str[0], 0
            # 通过统计每个元素出现的个数来创建输出
            for ch in pre_str:
                if ch == cur_val:
                    count_val += 1
                else:
                    all_val += str(count_val) + str(cur_val)
                    cur_val, count_val = ch, 1

            # 不能忘记添加尾项
            pre_str = all_val + str(count_val) + str(cur_val)

        return pre_str

    def lengthOfLastWord(self, s: str) -> int:
        # 最后一个单词需要省略空格，因此需要从后向前看
        """
        count = 0
        for ch in s:
            if ch == " ":
                count = 0
            else:
                count += 1

        return count

        # 从后向前看
        count, lens = 0, len(s)
        for i in range(lens-1, -1, -1):
            if s[i] == " ":
                # 注意读取了一个长度以后就需要结束了
                if count:
                    break
                count = 0
            else:
                count += 1
        return count
        """
        # 库函数调用
        return len(s.strip(' ').split(' ')[-1])

    def addBinary(self, a: str, b: str) -> str:
        # 可直接使用字符串与数字之间的关系进行转换计算
        # return str(bin(int(a, 2) + int(b, 2)))[2:]

        # 常规做法：翻转逐一比对
        len_a, len_b = len(a), len(b)
        dicts = {"0": 0, "1": 1}

        a, b = a[::-1], b[::-1]
        # 防御式做法：预先将字符串设为等长就能大大减少后续的处理流程
        if len_a < len_b:
            a += "0" * (len_b - len_a)
        else:
            b += "0" * (len_a - len_b)
            # 为后面的长度选择做准备
            len_b = len_a

        incre, res = 0, ""
        # 比对
        for i in range(len_b):
            cur_val = dicts[a[i]] + dicts[b[i]] + incre
            if cur_val > 1:
                incre = 1
                res += str(cur_val - 2)
            else:
                res += str(cur_val)
                incre = 0

        # 注意最后还有可能进位
        return (res + "1")[::-1] if incre else res[::-1]

    def isPalindrome(self, s: str) -> bool:
        # 先从中筛选， 去掉无用的元素
        pre_sort = ""
        for ch in s:
            if "0" <= ch <= "9" or "a" <= ch <= "z":
                pre_sort += ch
            elif "A" <= ch <= "Z":
                pre_sort += ch.lower()

        lens = len(pre_sort)

        # 慢在此处库函数的调用，如果采用双指针应该会快很多
        return pre_sort[:lens // 2] == pre_sort[lens - lens // 2:][::-1]

    def reverseString(self, s: List[str]) -> None:
        head, tail = 0, len(s) - 1
        # 设置双指针交换即可
        while head < tail:
            s[head], s[tail] = s[tail], s[head]
            head += 1
            tail -= 1

    def reverseVowels(self, s: str) -> str:
        # 想法是从两边提取元素，分别置于左右两个字符串，最后将其合并即可
        # 代码冗余度太高
        """
        vow, ans_left, ans_right = set('aeiouAEIOU'), "", ""
        head, tail = 0, len(s)-1
        if tail < 1:
            return s
        while head <= tail:
            if s[head] in vow:
                while tail > head and s[tail] not in vow:
                    ans_right = s[tail] + ans_right
                    tail -= 1
                ans_left += s[tail]
                # 注意到当两指针位置相同时，只需要放入一个即可
                if tail != head:
                    ans_right = s[head] + ans_right

            elif s[tail] in vow:
                while head < tail and s[head] not in vow:
                    ans_left += s[head]
                    head += 1
                ans_left += s[tail]
                if tail != head:
                    ans_right = s[head] + ans_right
            else:
                ans_left += s[head]
                # 应对奇数数量的字符串时的操作
                if tail != head:
                    ans_right = s[tail] + ans_right

            head += 1
            tail -= 1

        return ans_left + ans_right
        """
        # 思想是先找到需要置换的元素，在反转后逐一插入新字符串即可，需要遍历两次，但是中间的判断减少了很多
        # 在时间与代码量上完爆前面一种方法
        vow, res = set('aeiouAEIOU'), ""
        exch, pos = [x for x in s if x in vow][::-1], 0

        for ch in s:
            if ch in vow:
                res += exch[pos]
                pos += 1
            else:
                res += ch

        return res

    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        # use general method
        """
        dicts = {}
        # 先保存字典
        for ch in magazine:
            if ch in dicts:
                dicts[ch] += 1
            else:
                dicts[ch] = 1

        for ch in ransomNote:
            # 逐一进行查找
            if ch not in dicts or dicts[ch] < 1:
                return False
            dicts[ch] -= 1

        return True
        """
        # 使用 count 方法，效率会高很多
        coll = set(ransomNote)
        for ch in coll:
            if ransomNote.count(ch) > magazine.count(ch):
                return False

        return True

    def firstUniqChar(self, s: str) -> int:
        # 切记python里面的set返回的顺序是不确定的，因此在进行顺序判断时不能直接使用
        coll, rep = set(s), []
        # 统计出现的位置
        for ch in coll:
            if s.count(ch) == 1:
                rep.append(s.index(ch))

        # 返回最小值或默认-1
        return min(rep) if len(rep) else -1

    def addStrings(self, num1: str, num2: str) -> str:
        # 使用字典来映射关系
        dicts = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9}
        len_1, len_2 = len(num1), len(num2)

        # 取反方便计算机顺序遍历
        num1, num2 = num1[::-1], num2[::-1]
        # 将两字符串设置为等长方便处理
        if len_1 > len_2:
            num2 += "0" * (len_1 - len_2)
        else:
            num1 += "0" * (len_2 - len_1)
            len_1 = len_2

        incre, res = 0, ""
        # 逐一相加并进行条件判断
        for pos in range(len_1):
            cur_ans = dicts[num1[pos]] + dicts[num2[pos]] + incre

            if cur_ans > 9:
                res += str(cur_ans - 10)
                incre = 1
            else:
                res += str(cur_ans)
                incre = 0

        # 注意最后的增量判断
        return (res + "1")[::-1] if incre else res[::-1]

    def countSegments(self, s: str) -> int:
        # 调用库函数切分，并且除去长度为0的部分，即为结果
        return len([x for x in s.split(" ") if len(x)])
        # 也可以转化为数空格的个数

    def compress(self, chars: List[str]) -> int:
        # 需要细分考虑尾数，有点麻烦
        lens = len(chars)
        # 当长度为1时，排除考虑
        if lens == 1:
            return lens
        # 需要记录当前的统计位置，当前值以及其数量，以及用于尾数判断的标志
        count_pos, cur_count, cur_val, flag = 0, 1, chars[0], chars[-2]
        for i in range(1, lens):
            if chars[i] == cur_val:
                cur_count += 1
            # 此处不包含尾数与前一位相同的情况
            if chars[i] != cur_val or i == lens - 1:
                if cur_count > 1:
                    chars[count_pos], strs = cur_val, str(cur_count)
                    count_pos += 1
                    # 逐字符录入
                    for k in range(len(strs)):
                        chars[count_pos] = strs[k]
                        count_pos += 1
                else:
                    # 单一字符仅需打印本身
                    chars[count_pos] = cur_val
                    count_pos += 1

                cur_val, cur_count = chars[i], 1

        # 若统计位置已经超出长度，说明已经完成，否则需要考虑尾数
        if count_pos < lens and cur_val != flag:
            chars[count_pos] = cur_val
            count_pos += 1
        return count_pos

    def repeatedSubstringPattern(self, s: str) -> bool:
        # 常规思路
        """
        lens = len(s)
        # 缩小一半加上非倍数的开销
        for i in range(1, lens//2+1):
            # 判断是否被除尽
            if not lens % i and s[:i]*(lens // i) == s:
                return True

        return False
        """
        # 由于母串可由子串构成，因此必满足周期性结构(满满的数学)
        # 由此可有 假设由 n 个子串构成， 则2S = 2n， 去掉首尾后破坏两组余下 2n-2 组
        # 如果其中不包括 n 组子串，则有 2n-2 > n, 即 n < 2 , n = 1, 为非周期结构
        # 因此在两组去掉首尾的母串中必含一组母串
        return (s + s)[1:-1].find(s) != -1

    def detectCapitalUse(self, word: str) -> bool:
        # 直接进行判断
        """
        return True if len(word) < 2 or word.upper() == word or word.lower() == word or (word[0].upper() == word[0] and word[1:] == word[1:].lower()) else False
        """
        # 网上大神的逆向思维很好：从后向前看，如果是大写，则前面必全为大写，否则，第二个到最后的必须全为小写，其他情况皆为False
        if word[-1].upper() == word[-1]:
            if word[:-1].upper() == word[:-1]:
                return True
        else:
            # 考虑长度为1的情况
            if len(word) == 1 or word[1:].lower() == word[1:]:
                return True
        return False

    def findLUSlength(self, a: str, b: str) -> int:
        # 自身长度也算，也是服气
        return -1 if a == b else max(len(a), len(b))

    def reverseStr(self, s: str, k: int) -> str:
        lens, res = len(s), ""
        # times 保存处理次数，reside 保存余数
        times, reside = lens // (2 * k), lens % (2 * k)
        for i in range(times):
            res += s[2 * i * k:2 * i * k + k][::-1] + s[2 * i * k + k: 2 * (i + 1) * k]

        # 条件判断
        if reside >= k:
            res += s[2 * times * k:2 * times * k + k][::-1] + s[2 * times * k + k:]
        else:
            res += s[2 * times * k:][::-1]

        return res

    def checkRecord(self, s: str) -> bool:
        dicts = {"A": -1, "P": 0}
        # score 统计当前分数判断A与P， times统计迟到次数
        lens, score, times, pos = len(s), 1, 0, 0
        while pos < lens and score > -1 and times < 3:
            if s[pos] not in dicts:
                times += 1
            else:
                # 迟到必须为连续
                times = 0
                score += dicts[s[pos]]
            pos += 1

        return True if score > -1 and times < 3 else False

    def reverseWords(self, s: str) -> str:
        # 先整体反转获取各翻转字符串，再逆序输出即可
        """return " ".join([x for x in s[::-1].split(" ")][::-1])"""
        # 方法二：直接逐一逆序输出
        return " ".join(x[::-1] for x in s.split(" "))

    def initial_tree(self, arr: List[int]) -> TreeNode:
        # 由于None与类中的None的不同，因此只能单独设置
        def create(arrs: List[int], index: int):
            # 递归构建二叉树的方法：先序构造
            if index < len(arrs):
                Tn = TreeNode(None)
                Tn.val = arrs[index]
                Tn.left = create(arrs, index * 2 + 1)
                Tn.right = create(arrs, index * 2 + 2)
                return Tn
            else:
                return None

        return create(arr, 0)

    def tree2str(self, t: TreeNode) -> str:
        if not t:
            return ''
        """
        left = '(' + self.tree2str(t.left) + ')' if (t.left or t.right) else ''
        right = '(' + self.tree2str(t.right) + ')' if t.right else ''
        return str(t.val) + left + right
        """
        # 注意括号添加的条件：左子树添加为左不空或者右不空
        # 右子树则只能为右不空时添加
        res = str(t.val)
        if t.left or t.right:
            res += "(" + self.tree2str(t.left) + ")"
        if t.right:
            res += "(" + self.tree2str(t.right) + ")"

        return res

    def judgeCircle(self, moves: str) -> bool:
        # 字典法1慢出天际
        """
        dicts = {"R": [0, 1], "L": [0, -1], "U": [-1, 0], "D": [1, 0]}
        init = [0, 0]
        for ch in moves:
            init[0] += dicts[ch][0]
            init[1] += dicts[ch][1]
        return True if init == [0, 0] else False
        """
        # 字典法2优化了许多
        """
        dicts = {"L": -1, "R": 1, "U": -100, "D": 100}
        res = 0
        for ch in moves:
            res += dicts[ch]

        return False if res else True
        """
        # 内库调用最快
        return True if moves.count("L") == moves.count("R") and moves.count("U") == moves.count("D") else False

    def validPalindrome(self, s: str) -> bool:
        # 完全不用内库慢成go
        """
        st, ed, flag, born = 0, len(s)-1, True, False
        while st < ed:
            if s[st] != s[ed]:
                break
            st += 1
            ed -= 1

        def judge(s: str, st: int, ed: int):
            while st < ed:
                if s[st] != s[ed]:
                    return False
                st += 1
                ed -= 1
            return True

        return judge(s, st+1, ed) or judge(s, st, ed-1)
        """
        # 初始判断能节约大部分时间
        if s == s[::-1]:
            return True
        # 使用内库
        st, ed = 0, len(s) - 1
        while st < ed:
            if s[st] != s[ed]:
                break
            st += 1
            ed -= 1

        # 关于回文数的判断，直接判断是否逆转后仍然相等即可，无需求取中间值，学到了
        # 两种情况分别作判断即可
        a, b = s[st + 1: ed + 1], s[st:ed]
        return a == a[::-1] or b == b[::-1]

    def repeatedStringMatch(self, A: str, B: str) -> int:
        # 最多向两个方向延伸，因此最多加2
        times = len(B) // len(A) + 2
        for i in range(1, times + 1):
            if B in A * i:
                return i

        return -1

    def countBinarySubstrings(self, s: str) -> int:
        # 逐一统计果然会超时
        """
        total, times = 0, len(s) // 2
        cur_str = "01"
        for i in range(times):
            total += s.count(cur_str) + s.count(cur_str[::-1])
            cur_str = "0" + cur_str + "1"

        return total
        """
        # 分成两步来看待，分别为当前统计的数量与上一次统计数量，求取其最小值即可
        i, lens = 0, len(s)
        count = 0
        zero, one = 0, 0
        while i < lens:
            # 将先前变量保存
            pre = s[i]
            # 统计当前的数量
            while i < lens and pre == s[i]:
                zero += 1
                i += 1
            count += min(one, zero)
            one, zero = zero, 0

        return count

    def toLowerCase(self, str: str) -> str:
        res = ""
        for ch in str:
            if "A" <= ch <= "Z":
                # 普通转化
                res += chr(ord(ch) + 32)
            else:
                res += ch

        return res

    def rotatedDigits(self, N: int) -> int:
        """
        count = 0
        dicts = {0: 0, 1: 1, 2: 5, 5: 2, 6: 9, 8: 8, 9: 6}

        # 按照题给思路逐一比对的方法
        for i in range(1, N+1):
            cur_val, flag, dig, t = i, True, 0, 0
            while cur_val:
                # 整数的拆分
                div, mod = cur_val // 10, cur_val % 10
                if mod not in dicts:
                    flag = False
                    break
                cur_val = div
                # 比较变换后的数字
                dig += dicts[mod]*10**t
                t += 1

            if flag and i != dig:
                count += 1

        return count
        """
        # 方法二：直接判断，不用计算出结果
        count = 0
        for i in range(1, N + 1):
            cur = str(i)
            if "3" in cur or "4" in cur or "7" in cur:
                continue
            if "2" in cur or "5" in cur or "6" in cur or "9" in cur:
                count += 1
        return count

    def uniqueMorseRepresentations(self, words: List[str]) -> int:
        exch = [".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..", "--", "-.", "---",
                ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--.."]
        # 添加到set中自动虑重
        res = set()
        for strs in words:
            """
            cur_str = ""
            for ch in strs:
                # 此处应该可以简写
                cur_str += exch[ord(ch)-97]
            res.add(cur_str)
            """
            # 简写形式
            res.add("".join(map(lambda x: exch[ord(x) - 97], strs)))
        return len(res)

    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        # 无法过滤其它连接符号
        # lists = list(map(lambda x: x.lower().strip("!?.;,'"), paragraph.split()))

        # 将过滤集合转化为set，缩短时间
        ban, sym, cur_str = set(banned), set("!?.,'; "), ""
        # 添加空集
        ban.add('')
        dicts = {}
        # 由于中间连接词不止一种，只能采用逐一过滤，或者可以改为正则表达式
        for ch in paragraph:
            if ch not in sym:
                cur_str += ch.lower()
            else:
                if cur_str not in ban:
                    if cur_str in dicts:
                        dicts[cur_str] += 1
                    else:
                        dicts[cur_str] = 1
                cur_str = ""

        # 如果只有一个元素时，上面的循环根本不会判断
        if not len(dicts) and len(cur_str):
            return cur_str if cur_str not in ban else ""

        max_key, max_val = 0, ""
        # 找到出现次数最多的单词
        for key in dicts:
            if dicts[key] > max_key:
                max_val = key
                max_key = dicts[key]

        return max_val

    def toGoatLatin(self, S: str) -> str:
        judge, s = set("aeiouAEIOU"), S.split()
        res, lens = "", len(s)

        # 区分元音辅音即可
        for i in range(lens):
            if s[i][0] in judge:
                res += s[i] + "ma" + "a" * (i + 1) + " "
            else:
                cen_val = s[i][0] + "ma" + "a" * (i + 1) + " "

                # 注意辅音判别时必须要看长度，否则可能下标越界
                if len(s[i]) > 1:
                    res += s[i][1:] + cen_val
                else:
                    res += cen_val

        # 末尾空格删除
        return res[:-1]

    def buddyStrings(self, A: str, B: str) -> bool:
        # 唯一相同也满足的情况: error 还有其他情况
        len_a, len_b = len(A), len(B)
        if len_a == len_b:
            # 相同时必须满足串中至少包含两个相同字符，所以用set长度判断即可
            if len_a > len(set(A)) and A == B:
                return True
            # 双指针法确定位置
            st, ed = 0, len_a - 1
            while st < ed and A[st] == B[st]:
                st += 1
            while st < ed and A[ed] == B[ed]:
                ed -= 1
            # 剔除其他情况
            if st < ed and A[st] == B[ed] and A[ed] == B[st]:
                return True

        return False

    def numSpecialEquivGroups(self, A: List[str]) -> int:
        # 题意是通过奇偶交换使得两个字符串相同即为一组
        set_val = set()
        for val in A:
            # 先对元素进行奇偶排序，然后放入set中自动过滤重复
            set_val.add(str(sorted(val[::2])+sorted(val[1::2])))

        return len(set_val)

    def reverseOnlyLetters(self, S: str) -> str:
        # 同时加减有点复杂，还是需要存储
        """
        res, lens, k = "", len(S), 0
        pos_other, pos_al = 0, lens-1
        while pos_al > -1:
            while pos_other < lens and 97 <= ord(S[pos_other].lower()) <= 122:
                pos_other += 1

            while k < pos_other:
                if 97 <= ord(S[pos_al].lower()) <= 122:
                    res += S[pos_al]
                    k += 1
                pos_al -= 1

            if pos_other < lens:
                res += S[pos_other]
                pos_other += 1

        return res
        """
        # 防御式编程思想，无需在后面进行多余的判断
        S += "!"
        # 找出符号位置
        sym_pos = [pos for pos, ch in enumerate(S) if ord(ch.lower()) < 97]
        res = ""
        # cur_pos为逆序字母位置，i为字符位置下标，k为已经放置的数量
        cur_pos, i, k = len(S)-1, 0, 0
        while i < len(sym_pos):
            while k < sym_pos[i]:
                if 97 <= ord(S[cur_pos].lower()) <= 122:
                    res += S[cur_pos]
                    k += 1
                cur_pos -= 1

            # 注意在添加了非字符后，当前长度也得增加
            res += S[sym_pos[i]]
            k += 1
            i += 1

        # 最后一个非字母切除即可
        return res[:-1]

    def isLongPressedName(self, name: str, typed: str) -> bool:
        # 无字母顺序不一致时的情况
        """
        from collections import defaultdict
        dicts = defaultdict(int)
        for ch in typed:
            dicts[ch] += 1

        for ch in name:
            if ch not in dicts or dicts[ch] == 0:
                return False
            dicts[ch] -= 1

        return True
        """
        # 分别记录name下标，以及上一次的字符
        pos_n, lens, pre_val = 0, len(typed), name[0]
        for pos in range(lens):
            # 逐一比对
            if name[pos_n] == typed[pos]:
                pre_val = name[pos_n]
                pos_n += 1

            # 过滤重复元素
            elif typed[pos] != pre_val:
                return False

            # 必须在内部判断，否则可能会下标越界
            if pos_n == len(name):
                return True

        return False

    def numUniqueEmails(self, emails: List[str]) -> int:
        collect = set()
        for email in emails:
            cur_str = ""
            for ch in email:
                if ch == ".":
                    continue
                # 注意有可能直接遍历到@，因此也需要其作为终止符
                elif ch == "+" or ch == "@":
                    break
                cur_str += ch

            # 由于每个邮箱地址仅有一个@，因此可以使用index来寻找下标
            collect.add(cur_str+email[email.index("@"):])

        return len(collect)

    def reorderLogFiles(self, logs: List[str]) -> List[str]:
        # 未知错误：'o'为什么比'w'还排在后面？
        # 测试案例：["j mo", "5 m w", "g 07", "o 2 0", "t q h"]
        # 案例结果：["5 m w","j mo","t q h","g 07","o 2 0"]
        """
        alpha_list, digit_list = ["1 zzzzzz"], []

        def find_next(strs:str, pos_val: int):
            while pos_val < len(strs) and strs[pos_val].isspace():
                pos_val += 1
            return pos_val

        for log in logs:
            if log[-1].isdigit():
                digit_list.append(log)
            else:
                flag = False
                for i in range(len(alpha_list)):
                    pos_log, pos_cur = find_next(log, log.index(" ")+1), find_next(alpha_list[i], alpha_list[i].index(" ")+1)

                    # 用的是插入排序，由于python的list特性，不需要进行移项
                    while log[pos_log] == alpha_list[i][pos_cur]:
                        pos_log, pos_cur = find_next(log, pos_log+1), find_next(alpha_list[i], pos_cur+1)
                        # 特殊判断，当两者相同时
                        if pos_log == len(log) and pos_cur == len(alpha_list[i]):
                            flag = True
                            break

                    if flag:
                        alpha_list.insert(i+1, log) if log[:log.index(" ")] > alpha_list[i][:alpha_list[i].index(" ")] else alpha_list.insert(i, log)
                        break
                    # 防御式编程，设置一个极大数作为边界，因此无需进行后续判断
                    if log[pos_log] < alpha_list[i][pos_cur]:
                        alpha_list.insert(i, log)
                        break

        return alpha_list[:-1] + digit_list
        """
        nums = []
        tmp = []
        ret = []
        for k in logs:
            space = k.index(" ")
            if k[space + 1].isdigit():
                nums.append(k)
            else:
                mark = k[:space]
                tmp.append((mark, k[space + 1:]))
        # 利用了多重排序，但是没有去掉空格直接进行判断？
        for k in sorted(tmp, key=lambda x: (x[1], x[0])):
            ret.append(k[0] + " " + k[1])
        return ret + nums

    def gcdOfStrings(self, str1: str, str2: str) -> str:
        """
                min_str, max_str = str1, str2
                if len(str1) > len(str2):
                    min_str, max_str = max_str, min_str

                max_len = 0
                if max_str in 10*min_str:
                    for i in range(1, len(min_str)+1):
                        # 需要逐一遍历，太zz了
                        if not len(min_str) % i and min_str[:i]*(len(min_str)//i) == min_str and min_str[:i]*(len(max_str)//i) == max_str and i > max_len:
                            max_len = i

                return "" if not max_len else min_str[:max_len]
                """
        # 网上大神做法:模拟GCD 妙啊！
        while str1 != str2:
            if len(str1) < len(str2):
                str1, str2 = str2, str1
            # 不能整除完毕则必无最大公因子
            if str1[:len(str2)] != str2:
                return ""
            # 字符串无除法，因此此处用减法来慢慢削减，直至长度相同为止
            str1 = str1[len(str2):]
        return str1

    def defangIPaddr(self, address: str) -> str:
        # 使用join方法为每个中间项追加字符串
        return "[.]".join(address.split("."))

    def maxNumberOfBalloons(self, text: str) -> int:
        dicts = {"b": 0, "a": 0, "l": 0, "o": 0, "n": 0}
        for ch in text:
            if ch in dicts:
                dicts[ch] += 1
        # 应该取最大，但是找不到
        min_count = 100000
        for ch in dicts:
            # 有两种字符需要单独判别，取其中的最小即可
            if ch == 'l' or ch == 'o':
                min_count = min(min_count, dicts[ch]//2)
            else:
                min_count = min(min_count, dicts[ch])

        return min_count

    def balancedStringSplit(self, s: str) -> int:
        # 贪心思想
        count, balance = 0, 0
        for ch in s:
            if ch == 'R':
                balance += 1
            else:
                balance -= 1
            # 碰到就是赚到
            if not balance:
                count += 1

        return count


if __name__ == '__main__':
    show = Solution()

    # 13 罗马数字转整数
    # print(show.romanToInt("LVIII"))

    # 14 最长公共前缀
    # print(show.longestCommonPrefix(["dog","racecar","car"]))

    # 20 有效的括号
    # print(show.isValid(")"))

    # 28 实现strStr()
    # print(show.strStr("hello", "ll"))

    # 38 报数
    # print(show.countAndSay(7))

    # 58 最后一个单词长度
    # print(show.lengthOfLastWord("dhello world"))

    # 67 二进制求和
    # print(show.addBinary("100", "110010"))

    # 125 验证回文串
    # print(show.isPalindrome("race a car"))

    # 344 反转字符串
    # print(show.reverseString(["H","a","n","a","h"]))

    # 345 反转字符串中的元音字母
    # print(show.reverseVowels("leetcode"))

    # 383 赎金信
    # print(show.canConstruct("aa", "aab"))

    # 387 字符串中的第一个唯一字符
    # print(show.firstUniqChar("leetcode"))

    # 415 字符串相加
    # print(show.addStrings("111", "999"))

    # 434 字符串中的单词数
    # print(show.countSegments("Hello, my   name is John"))

    # 443 压缩字符串
    # print(show.compress(["a","a"]))

    # 459 重复的子字符串
    # print(show.repeatedSubstringPattern("abab"))

    # 520 检测大写字母
    # print(show.detectCapitalUse("Flag"))

    # 521 最长特殊序列I
    # print(show.findLUSlength("aba", "abaac"))

    # 541 反转字符串II
    # print(show.reverseStr("abcdefg", 8))

    # 551 学生出勤记录I
    # print(show.checkRecord("PPALLLP"))

    # 557 反转字符串中的单词III
    # print(show.reverseWords("Let's take LeetCode contest"))

    # 606 根据二叉树创建字符串
    # print(show.tree2str(show.initial_tree([1,2,3,None,4])))

    # 657 机器人能否返回原点
    # print(show.judgeCircle("UD"))

    # 680 验证回文字符串II
    # print(show.validPalindrome("aguokepatgbnvfqmgmlcupuufxoohdfpgjdmysgvhmvffcnqxjjxqncffvmhvgsymdjgpfdhooxfuupuculmgmqfvnbgtapekouga"))

    # 686 重复叠加字符串匹配
    # print(show.repeatedStringMatch("abc", "dabcdab"))

    # 696 计数二进制子串
    # print(show.countBinarySubstrings("000111000"))

    # 709 转换成小写字母
    # print(show.toLowerCase("Hello"))

    # 788 旋转数字
    # print(show.rotatedDigits(10))

    # 804 唯一摩尔斯密码词
    # print(show.uniqueMorseRepresentations(["gin", "zen", "gig", "msg"]))

    # 819 最常见的单词
    # print(show.mostCommonWord("Bob", []))

    # 824 山羊拉丁文
    # print(show.toGoatLatin("I speak Goat Latin"))

    # 859 亲密字符串
    # print(show.buddyStrings("bacccc", "abcccc"))

    # 893 特殊等价字符串组：难在题意的理解上面
    # print(show.numSpecialEquivGroups(["abc","acb","bac","bca","cab","cba"]))

    # 917 仅仅反转字母
    # print(show.reverseOnlyLetters("Test1ng-Leet=code-Q!"))

    # 925 长按键入
    # print(show.isLongPressedName("alex", "alexxx"))

    # 929 独特的电子邮件地址
    # print(show.numUniqueEmails(["test.email+alex@leetcode.com", "test.email@leetcode.com"]))

    # 937 重新排列日志文件
    # print(show.reorderLogFiles(["j mo", "5 m w", "g 07", "o 2 0", "t q h"]))

    # 1071 字符串的最大公因子
    # print(show.gcdOfStrings("ABCABC", "ABC"))

    # 1108 IP地址无效化
    # print(show.defangIPaddr("255.100.50.0"))

    # 1189 气球的最大数量
    # print(show.maxNumberOfBalloons("loonbalxballpoon"))

    # 1221 分隔平衡字符串
    # print(show.balancedStringSplit("RLLLLRRRLR"))
