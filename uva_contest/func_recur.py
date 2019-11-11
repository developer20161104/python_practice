import sys
from functools import reduce

times = 1


# p96 4-2
def squares():
    global times
    circle = sys.stdin.readline()

    while circle:
        n = int(circle)
        m = int(sys.stdin.readline())

        v, h = [], []
        jud_sort = []
        # pre work
        for i in range(m):
            ch, fi, sec = sys.stdin.readline().strip().split(' ')
            if ch == 'H':
                h.append([int(fi), int(sec)])
            else:
                # 判断是否需要进行后续处理
                ele_fi, ele_sec = int(fi), int(sec)
                if [ele_sec, ele_fi] in h:
                    jud_sort.append([ele_fi, ele_sec])
                v.append([ele_sec, ele_fi])
        # print(jud_sort)
        tot_stat = [0 for x in range(n)]
        for i in jud_sort:
            cur_len = 1
            # 判断是否包含更长长度的正方形
            while h.__contains__([i[0], i[1] + cur_len - 1]):
                flag = False
                for t in range(cur_len):
                    # 判断上下（h）以及左右（v）
                    if (not h.__contains__([i[0], i[1] + t]) or not v.__contains__([i[0] + t, i[1]])) or \
                            (not h.__contains__([i[0] + cur_len, i[1] + t]) or not v.__contains__(
                                [i[0] + t, i[1] + cur_len])):
                        flag = True
                        break
                # 满足构成正方形的条件才能加入统计
                if not flag:
                    tot_stat[cur_len - 1] += 1
                cur_len += 1

        print('Problem # ', times)
        times += 1
        if sum(tot_stat) == 0:
            print('No completed squares can be found.')
        else:
            for i in range(n):
                if tot_stat[i] != 0:
                    print(tot_stat[i], ' square (s) of size ', i + 1)

        circle = sys.stdin.readline()
        print('**********************************')


# p97 4-4
def cube_painting():
    strs = sys.stdin.readline()
    while strs:
        exc_1, exc_2 = [], []

        for i in range(3):
            exc_1.append([strs[i], strs[5 - i]])
            exc_2.append([strs[i + 6], strs[11 - i]])

        # 直接剔除相关元素，最后如果全部剔除完毕，则必相等
        for i in range(3):
            if exc_1[i] in exc_2:
                exc_2.remove(exc_1[i])
            elif [exc_1[i][1], exc_1[i][0]] in exc_2:
                exc_2.remove([exc_1[i][1], exc_1[i][0]])

        if len(exc_2):
            print('FALSE')
        else:
            print('TRUE')
        strs = sys.stdin.readline()


get_num = [128, 192, 224, 240, 248, 252, 254, 255]


# p97 4-5
def network_ip():
    n = sys.stdin.readline()
    while n:
        n = int(n)
        elem = []
        for i in range(n):
            elem.append(list(map(int, sys.stdin.readline().strip('\n').split('.'))))

        # print(elem)
        short_pos = []
        # 获取相同值作为最小IP
        strs_1 = ''
        for i in range(4):
            dicts = {}
            for k in range(n):
                if elem[k][i] in dicts:
                    dicts[elem[k][i]] += 1
                else:
                    dicts[elem[k][i]] = 1
            # print(dicts)
            if len(dicts) != 1:
                # 通过迭代获取第一个有序的dict组
                short_pos = sorted(list(dicts.keys()))
                break
            strs_1 += str(list(dicts.keys())[0]) + '.'

        last = reduce(lambda x, y: x & y, short_pos)
        strs_1 += str(last)

        for k in range(3 - i):
            strs_1 += '.0'

        # 根据得到的最小IP求取子网掩码
        strs_2 = ''
        for t in range(i):
            strs_2 += '255.'

        strs_2 += str(get_num[bin(last).rfind('1') - 1])

        for k in range(3 - i):
            strs_2 += '.0'

        print(strs_1)
        print(strs_2)

        # use i to guess length
        n = sys.stdin.readline()


if __name__ == '__main__':
    # squares()
    '''
    h = [[1,2],[2,3],[3,4]]
    for i in h:
        print(i[0])
    '''
    # cube_painting()
    network_ip()
    # print(bin(get_num[bin(176).rfind('1')-2]))
