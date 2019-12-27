from typing import List


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
                y = y * 10 + x % 10
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
            reverse = reverse * 10 + x % 10
            x //= 10

        # 此处需要看源数字是否为奇数量还是偶数量
        return reverse == x or reverse // 10 == x

    def mySqrt(self, x: int) -> int:
        # 神奇的魔数 0x5f3759df
        s1 = x
        while abs(s1 * s1 - x) > 0.1:
            # 牛顿法x（n+1）=x（n）+ S/x（n）
            s1 = (s1 + x / s1) / 2

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
            res += (ord(ch) - 64) * (26 ** k)
            k += 1

        return res

    def trailingZeroes(self, n: int) -> int:
        # 通过暴力破解可以发现规律：将2与5进行配对，每出现一对则结果尾数必含一个0
        # 由于5出现的次数一定比2少，因此只需要判断5的个数即可，当然也要考虑25 等特殊情形
        # 类似条件表达式的递归写法
        return 0 if not n else n // 5 + Solution.trailingZeroes(self, n // 5)

    def isHappy(self, n: int) -> bool:
        # 关键点：保存当前已经出现过的数值以作为结束依据
        res_set = set()
        cur_val = 0
        while cur_val != 1:
            # 直接进行模拟操作
            while n:
                cur_val += (n % 10) ** 2
                n //= 10

            # 结束依据判断
            if cur_val in res_set:
                return False
            # 当前数值不满足条件时继续下一轮判断并保留计算结果
            if cur_val != 1:
                res_set.add(cur_val)
                n = cur_val
                cur_val = 0

        return True

    def countPrimes(self, n: int) -> int:
        # 暴力统计必然会超时
        """
        import math
        count = 0
        if n < 3:
            return 0
        for i in range(2, n):
            flag = True
            for k in range(2, int(math.sqrt(i))+1):
                if not i % k:
                    flag = False
                    break

            if flag:
                count += 1
        return count
        """
        # 需要用空间来换时间
        # 使用一个状态列表来保存所有的数字，通过质数的倍数来进行消除
        if n < 3:
            return 0

        state = [0] * n
        # 只计算根号前面部分即可
        l = int(n ** 0.5)

        for i in range(2, l + 1):
            if state[i]:
                continue
            # 此处将质数的倍数部分全部填满
            state[i * i:n:i] = [1] * len(state[i * i:n:i])

        # 注意要删减0,1两个数字
        return len(state) - sum(state) - 2

    def isPowerOfTwo(self, n: int) -> bool:
        # 需要考虑边界条件
        if n < 1:
            return False
        # 只要在求取中出现除1外的奇数，即为False
        while n != 1:
            if n % 2:
                return False
            n //= 2

        return True

    def addDigits(self, num: int) -> int:
        # 考虑到9时的特殊情况：妙啊！
        return (num - 1) % 9 + 1 if num else 0
        # 一种更快的方法：9的倍数各位加上的和还是9，错误，只是在样本范围内可以抓小空子
        # return num % 9 or 9 * bool(num)

    def isUgly(self, num: int) -> bool:
        if num < 1:
            return False
        # 需要判断的除数
        div = [2, 3, 5]

        while num != 1:
            res = num
            for div_d in div:
                # 只要包含就行
                if not res % div_d:
                    res = res // div_d
                    break
            # 如果不包含则直接返回False
            if res == num:
                return False
            # 最后替换下一轮元素
            num = res

        return True

    def isPowerOfThree(self, n: int) -> bool:
        # 常规循环解法
        """
        if n < 1:
            return False
        # 只要在求取中出现除1外的奇数，即为False
        while n != 1:
            if n % 3:
                return False
            n //= 3

        return True
        """
        # 找限制范围：直接使用取值的上界作为判断基准即可，limit:3**19
        return n > 0 and not 1162261467 % n

    def isPerfectSquare(self, num: int) -> bool:
        # 暴力法，本质为查找
        """
        if num < 1:
            return False
        cur = 1
        while cur**2 < num:
            cur += 1

        return True if cur**2 == num else False
        """
        # 思路为二分查找，但是列表的空间还能优化
        """
        if num < 4:
            return False if num != 1 else True

        arr = range(1, num//2+1)
        left, right = 0, len(arr)-1
        while left < right:
            mid = (left + right)//2
            if arr[mid]**2 < num:
                left = mid + 1
            else:
                right = mid

        return arr[left]**2 == num
        """
        # 优化的查找部分可以忽略不计 -_-!
        left, right = 1, num
        while left < right:
            mid = (left + right) // 2
            # 常规二分查找，顺序问题
            if mid ** 2 < num:
                left = mid + 1
            else:
                right = mid

        return left ** 2 == num

    def arrangeCoins(self, n: int) -> int:
        # 实质还是一个二分查找问题
        left, right = 1, n
        while left < right:
            mid = (left + right) // 2
            # 只是查找时比较的是等差求和公式(n+1)*n//2
            if mid * (mid + 1) // 2 < n:
                left = mid + 1
            else:
                right = mid

        # 由于采用整数截断，因此需要将刚好相等的进行特殊判断
        return left - 1 if n != left * (left + 1) // 2 else left

    def minMoves(self, nums: List[int]) -> int:
        # 整体思路不对，只用数学上的等式无法完全满足要求
        """
        if len(set(nums)) == 1:
            return 0
        lens, time, res = len(nums), 1, sum(nums)
        while (res + time*(lens-1)) % lens:
            time += 1

        return time
        """
        # 需要暴力查找规律
        compare, time = min(nums), 0
        for dig in nums:
            time += dig - compare

        return time

        # 虽然一行解决，但是过于耗时
        # return sum([x-min(nums) for x in nums])

    def checkPerfectNumber(self, num: int) -> bool:
        # 动态限制长度，还是会超时
        """
        res = set()
        cur, limit = 1, num
        while cur < limit:
            if not num % cur and cur not in res:
                res.add(cur)
                if cur != 1:
                    limit = num//cur
                    res.add(limit)

            cur += 1

        return sum(res) == num
        """
        # 方法依旧暴力，但是将上限限制到了sqrt(num)，因此不会超时
        if num < 1:
            return False
        import math
        total, limit = 0, int(math.sqrt(num))+1
        for i in range(1, limit):
            if not num % i:
                total += i
                if i*i != num:
                    total += num // i

        return total - num == num


if __name__ == '__main__':
    show = Solution()

    # 7 整数反转
    # print(show.reverse(-1234))

    # 9 回文数
    # print(show.isPalindrome(10))

    # 69 x的平方根
    # print(show.mySqrt(1000))

    # 168 Excel表列名称
    # print(show.convertToTitle(702))

    # 171 Excel表列序号
    # print(show.titleToNumber("AAA"))

    # 172 阶乘后的零
    # print(show.trailingZeroes(10))

    # 202 快乐数
    # print(show.isHappy(10))

    # 204 计数质数
    # print(show.countPrimes(10))

    # 231 2的幂
    # print(show.isPowerOfTwo(0))

    # 258 各位相加
    # print(show.addDigits(27))

    # 263 丑数
    # print(show.isUgly(1))

    # 326 3的幂
    # print(show.isPowerOfThree(3))

    # 367 有效的完全平方数
    # print(show.isPerfectSquare(43537405842735850251441))

    # 441 排列硬币
    # print(show.arrangeCoins(6))

    # 453 最小移动次数使得数组元素相等
    # print(show.minMoves([1, 2, 3]))

    # 507 完美数
    # print(show.checkPerfectNumber(25964951))
