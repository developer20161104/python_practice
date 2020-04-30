import random
from typing import List
import ExtendedEuclidean as ex


def Shamir_share(n: int, s: int, p: int) -> List[List[int]]:
    # 返回n个点的坐标
    # a_sort = [random.randint(Zp) for _ in range(1, k)]
    # 测试ppt案例
    a_sort = [2, 7]

    res = []
    for i in range(1, n + 1):
        cur_f = s
        for index, a in enumerate(a_sort):
            cur_f += a * (i ** (index + 1))
        res.append([i, cur_f % p])

    return res


def Shamir_reconstruct(points: List[List[int]], p: int) -> int:
    # 仅需要k个点坐标来重构函数f
    result = 0
    # 即为k
    lens = len(points)
    for i in range(lens):
        L_up, L_down = 1, 1
        for j in range(lens):
            # 由于只需要求取密钥，因此将x置零
            L_up *= -points[j][0]
            if i != j:
                L_down *= (points[i][0] - points[j][0])
        L_up //= -points[i][0]

        # 此处需要求分数的逆元，因此需要拓展欧几里得求取
        # result += L_up//(L_down**-1 % p)*points[i][1]
        result += L_up * (ex.EEa(L_down, p)[0]) * points[i][1]

    # 最后结果约束到Zp 范围
    return result % p


if __name__ == '__main__':
    # share阶段
    print(Shamir_share(5, 11, 19))
    # reconstruction阶段
    print(Shamir_reconstruct([[2, 5], [3, 4], [5, 6]], 19))
