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
        total, limit = 0, int(math.sqrt(num)) + 1
        for i in range(1, limit):
            if not num % i:
                total += i
                if i * i != num:
                    total += num // i

        return total - num == num

    def maxCount(self, m: int, n: int, ops: List[List[int]]) -> int:
        # 直接模拟会内存溢出
        """
        count = [0]*(m*n)
        for e in ops:
            for i in range(e[0]):
                count[i*n:i*n+e[1]] = [x+1 for x in count[i*n:i*n+e[1]]]

        return count.count(count[0])
        """
        # 短板效应:只需考虑最短的位置即可，已经限制了顶点位，因此可以使用此方法
        if not ops:
            return m * n
        # 将多维列表解压
        a, b = zip(*ops)
        return min(a) * min(b)

    def judgeSquareSum(self, c: int) -> bool:
        limit = c ** 0.5
        for i in range(1, int(limit) + 1):
            # 此处为判断求平方根是否为整数
            res = (c - i * i) ** 0.5
            if int(res) == res:
                return True

        return False if c else True

    def findErrorNums(self, nums: List[int]) -> List[int]:
        # 以空间换时间的做法
        # 题给案例限制，因此设置为n长度即可
        res = [0] * 10000
        for i in nums:
            res[i - 1] += 1

        # 找到出现两次与没有出现的元素
        return [res.index(2) + 1, res.index(0) + 1]

    def selfDividingNumbers(self, left: int, right: int) -> List[int]:
        res = []
        for i in range(left, right + 1):
            # 当小于10时，直接添加
            if i < 10:
                res.append(i)
            elif i % 10:
                # 保留原数
                jud = i
                while i:
                    div = i % 10
                    # 需要判断包含0的情况，直接返回
                    if not div or jud % div:
                        break
                    i = i // 10

                # 如果没有完全判断，则必不为自除数
                if not i:
                    res.append(jud)

        return res

    def largestTriangleArea(self, points: List[List[int]]) -> float:
        # 暴力法即可，需要使用海伦公式
        # 纯数学
        res, lens = 0, len(points)
        for i in range(lens):
            x1, y1 = points[i][0], points[i][1]
            for j in range(i + 1, lens):
                x2, y2 = points[j][0], points[j][1]

                for k in range(j + 1, lens):
                    x3, y3 = points[k][0], points[k][1]

                    # 海伦公式
                    res = max(res, abs(x1 * y2 + x2 * y3 + x3 * y1 - x2 * y1 - x3 * y2 - x1 * y3) / 2)

        return res

    def isRectangleOverlap(self, rec1: List[int], rec2: List[int]) -> bool:
        # 判断相交根本行不通，情况过多，无法考虑，因此要反其道行之
        """
        min_x, min_y, lar_x, lar_y = rec1[:]
        if min_x > lar_x:
            min_x, lar_x = lar_x, min_x
        if min_y > lar_y:
            min_y, lar_y = lar_y, min_y

        def judge(mins, maxs, pos):
            return mins < pos < maxs
        return True if (judge(min_x, lar_x, min(rec2[0], rec2[2])+0.1) and judge(min_y, lar_y, min(rec2[1], rec2[3])+0.1)) \
            or (judge(min_x, lar_x, max(rec2[0], rec2[2])-0.1) and judge(min_y, lar_y, max(rec2[1], rec2[3])-0.1)) else False
        """
        # 题目已经限定位置，前一个为左下，后一个为右上，不需要判断大小
        # 判断不在的情况，分别表示rec1在右侧 上侧 左侧 下侧
        return not (rec1[0] >= rec2[2] or rec1[1] >= rec2[3] or rec1[2] <= rec2[0] or rec1[3] <= rec2[1])

    def binaryGap(self, N: int) -> int:
        # 分别统计0出现的次数以及是否出现，注意只出现一个1时候的情况即可
        count, max_cou, time = 0, 0, False
        while N:
            div = N % 2
            if div:
                # 出现多个1时才会有此处使用
                if time:
                    # 在计算距离时需要再加上本身的距离
                    max_cou = max(max_cou, count + 1)
                time = True
                count = 0
            else:
                count += 1

            N //= 2

        return max(max_cou, count)

    def projectionArea(self, grid: List[List[int]]) -> int:
        # 俯视图对应元素数，侧视图对应行最长，正视图对应列最长（对于不等长的列表，无法使用zip，因此不考虑zip的解压）
        count = 0
        max_column = [0] * len(grid[0])
        for pos in grid:
            lens = len(pos)
            for i in range(lens):
                # 统计列最长
                max_column[i] = max(max_column[i], pos[i])
                if pos[i]:
                    # 统计元素个数
                    count += 1

            # 统计行最长
            count += max(pos)

        return count + sum(max_column)

    def surfaceArea(self, grid: List[List[int]]) -> int:
        # 需要对六个面进行计算
        # 转化为求行最长，列最长以及元素个数统计，不能忽略中间的凹槽部分面积
        # 整体统计完全不可行，会有重复
        surface, lens_row, len_col, back = 0, len(grid[0]), len(grid), 0

        # 分行列逐一判断
        for ro_cur in grid:
            # 边缘侧面积
            surface += ro_cur[0] + ro_cur[-1]
            # back保存的是底面积
            back += lens_row - ro_cur.count(0)
            for i in range(1, lens_row):
                # 逐一比对邻近块，横向侧面积
                surface += abs(ro_cur[i] - ro_cur[i - 1])
        for co_cur in zip(*grid):
            # 边缘正面积
            surface += co_cur[0] + co_cur[-1]
            for j in range(1, len_col):
                # 纵向正面积
                surface += abs(co_cur[j] - co_cur[j - 1])

        return surface + 2 * back

    def smallestRangeI(self, A: List[int], K: int) -> int:
        # 当两者差值K已经无法弥补了，则输出差值，否则一律为0
        return max(max(A) - min(A) - 2 * K, 0)

    def diStringMatch(self, S: str) -> List[int]:
        lens = len(S)
        cur_max, cur_min = lens, 0
        res = []

        for val in S:
            # 对于小于填写当前最小值即可
            if val == "I":
                res.append(cur_min)
                cur_min += 1
            # 对于大于则选择当前最大值
            else:
                res.append(cur_max)
                cur_max -= 1

        # 最后剩余的也需要添加上去
        res.append(cur_min)

        return res

    def largestTimeFromDigits(self, A: List[int]) -> str:
        res = -1
        # 暴力字典序
        for i in range(4):
            for j in range(4):
                if i != j:
                    for k in range(4):
                        if k != j and k != i:
                            l = 6 - i - j - k
                            hour, minute = A[i] * 10 + A[j], A[k] * 10 + A[l]
                            if hour < 24 and minute < 60:
                                res = max(res, hour * 60 + minute)

        # 格式化输出格式，少了要补零
        return "" if res < 0 else "{:02}:{:02}".format(res // 60, res % 60)

    def powerfulIntegers(self, x: int, y: int, bound: int) -> List[int]:
        """
        res = set()

        # 先逐一计算
        def getall(xs: int, bounds: int) -> List[int]:
            if xs == 1:
                return [1]
            p, rep = 0, 1
            reps = []
            while rep < bound:
                reps.append(rep)
                p += 1
                rep = xs**p

            return reps

        # 再合并结果
        ans1, ans2 = getall(x, bound), getall(y, bound)
        for k1 in ans1:
            for k2 in ans2:
                if k1 + k2 <= bound:
                    res.add(k1 + k2)

        return list(res)
        """
        # 各种边界条件有点恶心
        i, j = 0, 0
        if x == 1:
            if y == 1:
                # 还需要判断约束范围
                return [2] if bound > 1 else []
            x, y = y, x
        res = set()
        rep = 2
        while rep <= bound:
            while rep <= bound:
                res.add(rep)
                j += 1
                rep = x ** i + y ** j
                # 将x或y等于1的情况放内部处理，只需要保留一项即可
                if y == 1:
                    break
            j = 0
            i += 1
            rep = x ** i + y ** j

        return list(res)

    def largestPerimeter(self, A: List[int]) -> int:
        # 思路有错，不能只找当前最大的三个数
        """
        arr = A[:3]
        cur_max = min(arr)
        for i in A[3:]:
            if i > cur_max:
                arr[arr.index(cur_max)] = i
                cur_max = i

        return sum(arr) if sum(arr)-max(arr) > max(arr) else 0
        """
        # 先对总体排序，若要构成最长的周长，则每个元素在满足构成条件下寻找最优即可
        new_a = sorted(A)
        lens = len(new_a)
        for i in range(lens - 3, -1, -1):
            if new_a[i] + new_a[i + 1] > new_a[i + 2]:
                return sum(new_a[i:i + 3])

        # 找不到就返回0
        return 0

    def bitwiseComplement(self, N: int) -> int:
        i = 0
        # 注意可以相等
        while 2 ** i <= N:
            i += 1

        # 只含有一位时需要特别判断
        return N ^ (2 ** i - 1) if N > 1 else N ^ 1

    def isBoomerang(self, points: List[List[int]]) -> bool:
        # 可以优化，待会再改
        max_p, min_p = points[0], points[0]
        cur1, cur2 = 0, 0
        # 可能都不需要求解大小，直接进行判断即可
        for i in range(1, 3):
            if points[i] > max_p:
                max_p, cur1 = points[i], i
            if points[i] < min_p:
                min_p, cur2 = points[i], i
        if cur1 == cur2:
            return False
        k_f = max_p[0] - min_p[0]
        k = max_p[1] - min_p[1]

        cur3 = 3 - cur1 - cur2
        # 考虑k不存在时候的情况
        if k_f:
            k /= k_f
            b = max_p[1] - k * max_p[0]
            if k * points[cur3][0] + b != points[cur3][1]:
                return True
        else:
            if points[cur3][0] != max_p[0]:
                return True

        return False

    def distributeCandies(self, candies: int, num_people: int) -> List[int]:
        i = int(candies ** 0.5)
        # 先找到有多少项，取向上值
        while i * (i + 1) < 2 * candies:
            i += 1

        # 选取前i-1项再加上尾项
        rep = list(range(1, i)) + [candies - (i - 1) * i // 2]
        res = [0] * num_people

        for j in range(num_people):
            # 分间隔求和即可
            res[j] = sum(rep[j::num_people])

        return res

    def dayOfYear(self, date: str) -> int:
        # 列表保存每月的天数
        day_of_mon = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        # 切割年月日
        value = [int(v) for v in date.split('-')]
        day = value[2]

        # 当月份大于2月时再进行求和
        if value[1] > 1:
            day += sum(day_of_mon[:value[1] - 1])

        # 判断是否为闰年
        return day + 1 if value[1] > 2 and not value[0] % 4 and value[0] != 1900 else day


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

    # 598 范围求和II
    # print(show.maxCount(3, 3, [[2, 2], [3, 3]]))

    # 633 平方数之和
    # print(show.judgeSquareSum(20))

    # 645 错误的集合
    # print(show.findErrorNums([1,2,2,4]))

    # 728 自除数
    # print(show.selfDividingNumbers(1, 120))

    # 812 最大三角形面积
    # print(show.largestTriangleArea([[0,0],[0,1],[1,0],[0,2],[2,0]]))

    # 836 矩形重叠
    # print(show.isRectangleOverlap([2,17,6,20],[3,8,6,20]))

    # 868 二进制间距
    # print(show.binaryGap(8))

    # 883 三维形体投影面积
    # print(show.projectionArea([[2,2,2],[2,1,2],[2,2,2]]))

    # 892 三维形体的表面积
    # print(show.surfaceArea([[3,3,3],[3,4,5],[5,0,4]]))

    # 908 最小差值I
    # print(show.smallestRangeI([1,3,6],3))

    # 942 增减字符串匹配
    # print(show.diStringMatch("DDI"))

    # 949 给定数字能组成的最大时间
    # print(show.largestTimeFromDigits([1, 2, 3, 4]))

    # 970 强整数
    # print(show.powerfulIntegers(1,2,10))

    # 976 三角形的最大周长
    # print(show.largestPerimeter([2,2,1]))

    # 1009 十进制整数的反码
    # print(show.bitwiseComplement(2))

    # 1037 有效的回旋镖
    # print(show.isBoomerang([[0,2],[0,1],[0,1]]))

    # 1103 分糖果II
    # print(show.distributeCandies(22,3))

    # 1154 一年中的几天
    # print(show.dayOfYear("2019-02-10"))
