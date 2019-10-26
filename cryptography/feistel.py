# 密钥集与S盒，S盒使用的是DES中的S1盒
r_key = ('111111', '111000', '000111', '000000')

s_box = ((14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7),
         (0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8),
         (4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0),
         (15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13))


def f_func(r_byte: str, times: int) -> str:
    # 填充操作
    E_r = int(r_byte[1] + r_byte + r_byte[2], 2)
    # 与key异或并将结果限制为6位二进制字符串输出
    pre_s = '{0:06b}'.format(E_r ^ int(r_key[times], 2))

    # S盒操作
    s_row, s_col = 2 * int(pre_s[0], 2) + int(pre_s[5], 2), int(pre_s[1:5], 2)

    # P盒操作直接返回结果
    pre_p = '{0:04b}'.format(s_box[s_row][s_col])
    return pre_p[1] + pre_p[3] + pre_p[0] + pre_p[2]


def feistel_struct(name: str) -> str:
    # 保存二进制格式的姓名，加密后的姓名，以及Unicode编码后的姓名
    name_bin = ''
    code_bin = ''
    code = ''

    len_name = len(name)
    # 由于只能逐字符转化，因此只能逐一保存
    for i in range(len_name):
        name_bin += '{0:016b}'.format(ord(name[i]))

    times_cut_name = len(name_bin) // 8

    for i in range(times_cut_name):
        # 每次选取8位作为输入
        cur_sort = name_bin[i * 8:(i + 1) * 8]

        # 采用Feistel结构，一共轮转4次
        for j in range(4):
            next_L = cur_sort[4:8]
            next_R = int(f_func(cur_sort[4:8], j), 2) ^ int(cur_sort[0:4], 2)
            cur_sort = next_L + '{0:04b}'.format(next_R)

        code_bin += cur_sort

    print('加密后的密文为： ', code_bin)

    for i in range(len_name):
        # 输出加密后的汉字转化的数字
        # print(int(code_bin[i*16:(i+1)*16], 2))
        # 转化为字符
        code += chr(int(code_bin[i * 16:(i + 1) * 16], 2))
    return code


if __name__ == '__main__':
    # 汉字范围
    # print('汉字范围为： ', int('0x4e00', 16), ' to ', int('0x9fa6', 16))
    print('经过Unicode编码后得到的密文结果为： ', feistel_struct('杨斌'))
