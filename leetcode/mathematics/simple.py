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


if __name__ == '__main__':
    show = Solution()

    # 7 整数反转
    print(show.reverse(-1234))

    # 9 回文数
    print(show.isPalindrome(10))