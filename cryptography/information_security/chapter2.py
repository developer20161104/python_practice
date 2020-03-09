def mod_suqare(b: int, time: int, mod_num: int) -> int:
    n_sort, time_cal = [], time

    # 获取ni
    while time_cal:
        n_sort.append(time_cal % 2)
        time_cal //= 2

    a = 1
    for cur_n in n_sort:
        # 如果为1则计算a*b，否则保留当前a不变
        if cur_n:
            a = (a * b) % mod_num
        # b 每次都做平方计算
        b = (b**2) % mod_num

    return a


if __name__ == '__main__':
    # 模重复平均法
    print(mod_suqare(12996, 227, 37909))

    print(mod_suqare(163, 237, 667))
