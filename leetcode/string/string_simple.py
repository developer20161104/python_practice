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


if __name__ == '__main__':
    show = Solution()

    # 13 罗马数字转整数
    # print(show.romanToInt("LVIII"))

    # 14 最长公共前缀
    # print(show.longestCommonPrefix(["dog","racecar","car"]))
