from typing import List


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
            if sp+1 < lens and s[sp:sp+2] in special:
                count += special[s[sp:sp+2]]
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
        for _ in range(2, n+1):
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
            a += "0"*(len_b - len_a)
        else:
            b += "0"*(len_a - len_b)
            # 为后面的长度选择做准备
            len_b = len_a

        incre, res = 0, ""
        # 比对
        for i in range(len_b):
            cur_val = dicts[a[i]] + dicts[b[i]] + incre
            if cur_val > 1:
                incre = 1
                res += str(cur_val-2)
            else:
                res += str(cur_val)
                incre = 0

        # 注意最后还有可能进位
        return (res+"1")[::-1] if incre else res[::-1]

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
        return pre_sort[:lens//2] == pre_sort[lens-lens//2:][::-1]

    def reverseString(self, s: List[str]) -> None:
        head, tail = 0, len(s)-1
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
        return (res+"1")[::-1] if incre else res[::-1]

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
            if chars[i] != cur_val or i == lens-1:
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
        times, reside = lens // (2*k), lens % (2*k)
        for i in range(times):
            res += s[2*i*k:2*i*k+k][::-1] + s[2*i*k+k: 2*(i+1)*k]

        # 条件判断
        if reside >= k:
            res += s[2*times*k:2*times*k+k][::-1] + s[2*times*k+k:]
        else:
            res += s[2*times*k:][::-1]

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
