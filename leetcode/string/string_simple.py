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
