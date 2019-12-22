import math


class Solution:
    def reverse(self, x: int) -> int:
        """
        # 调用了库函数进行转换
        res = str(x)[::-1]
        # 只需要进行正负判断即可
        if res[-1] == "-":
            res = -int(res[:-1])
        else:
            res = int(res)

        return res if -2**31 <= res <= 2**31-1 else 0
        """
        # 取反还能这么玩
        y = 0
        while x:
            # 由于数字大小的限制，因此满足此项即可
            if y > 214748364 or y < -214748364:
                return 0

            # 数字取反的新套路
            # python 对负数取余时需要使用负数来求取
            if x > 0:
                y = y*10 + x % 10
            else:
                y = y * 10 + x % -10

            # python 特性，在对负数操作时，只能通过强制类型转换来进行修改
            x = int(x / 10)

        return y

    def isPalindrome(self, x: int) -> bool:
        # 通过求取反转数字进行判别，无需转换为字符串，但是需要全部转化
        """
        if x < 0:
            return False
        temp = x
        y = 0
        while temp:
            y = y*10 + temp % 10
            temp = temp // 10

        return True if y == x else False
        """
        # 转化一半即可, 节约了一半时间
        if x < 0 or (x and not x % 10):
            return False

        reverse = 0
        while x > reverse:
            reverse = reverse*10 + x % 10
            x //= 10

        # 此处需要看源数字是否为奇数量还是偶数量
        return reverse == x or reverse//10 == x

    def mySqrt(self, x: int) -> int:
        # 神奇的魔数 0x5f3759df
        s1 = x
        while abs(s1*s1 - x) > 0.1:
            # 牛顿法x（n+1）=x（n）+ S/x（n）
            s1 = (s1 + x/s1)/2

        return int(s1)

    def convertToTitle(self, n: int) -> str:
        res = ""
        while n > 0:
            y = n % 26
            # 难点主要在于对求余时没有对应的结果，因此需要进行手动转化
            if not y:
                # 从商中借一个拿来替换当前的余数
                n -= 1
                y = 26
            res = chr(64 + y) + res
            n //= 26

        return res

    def titleToNumber(self, s: str) -> int:
        res, k = 0, 0
        for ch in s[::-1]:
            # 简单的进制转化
            res += (ord(ch)-64)*(26**k)
            k += 1

        return res

    def trailingZeroes(self, n: int) -> int:
        # 通过暴力破解可以发现规律：将2与5进行配对，每出现一对则结果尾数必含一个0
        # 由于5出现的次数一定比2少，因此只需要判断5的个数即可，当然也要考虑25 等特殊情形
        # 类似条件表达式的递归写法
        return 0 if not n else n//5 + Solution.trailingZeroes(self, n // 5)


if __name__ == '__main__':
    show = Solution()

    # 7 整数反转
    print(show.reverse(-1234))

    # 9 回文数
    print(show.isPalindrome(10))

    # 69 x的平方根
    print(show.mySqrt(1000))

    # 168 Excel表列名称
    print(show.convertToTitle(702))

    # 171 Excel表列序号
    print(show.titleToNumber("AAA"))

    # 172 阶乘后的零
    print(show.trailingZeroes(10))