from typing import List


# s*a+t*b = (a, b)
def EEa(a: int, b: int) -> List[int]:
    # s,t 分别为因子
    pre_s, s = 1, 0
    pre_t, t = 0, 1
    # r为余数
    pre_r, r = a, b
    if not b:
        # 输出的为前一位
        # 分别为s，t，a
        return [1, 0, a]
    while r:
        # q 为商
        q = pre_r // r

        # 主要递推式
        # Si+1 = Si-1 - q* Si
        pre_r, r = r, pre_r-q*r
        pre_s, s = s, pre_s-q*s
        pre_t, t = t, pre_t-q*t

    return [pre_s, pre_t, pre_r]


if __name__ == '__main__':
    print(EEa(240, 46))