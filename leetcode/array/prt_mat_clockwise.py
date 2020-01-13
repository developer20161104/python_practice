from typing import List


def print_clockwise(mat: List[List[int]]):
    if not mat:
        return
    # 遍历顺序，右 下 左 上
    seq_sort = [[0, 1], [1, 0], [0, -1], [-1, 0]]

    # 当前位置
    cur_pos = [0, 0]
    # 当前方向
    seq = 0
    # 行列宽度
    len_r, len_c = len(mat), len(mat[0])
    while mat[cur_pos[0]][cur_pos[1]] != -1:
        # 打印当前位置
        print(mat[cur_pos[0]][cur_pos[1]], end='\t')
        # 置为已访问，减去visit列表构建开销
        mat[cur_pos[0]][cur_pos[1]] = -1

        # 先预判
        temp1, temp2 = cur_pos[0] + seq_sort[seq][0], cur_pos[1] + seq_sort[seq][1]
        if not (0 <= temp1 < len_r
                and 0 <= temp2 < len_c) or mat[temp1][temp2] == -1:
            seq = (seq + 1) % 4
            if len_r == 1:
                break

        # 再更改位置
        cur_pos[0] += seq_sort[seq][0]
        cur_pos[1] += seq_sort[seq][1]


def create_mat(row: int, col: int) -> List[List[int]]:
    # 创建二维列表
    return [list(range(1, row*col+1))[i*col:i*col+col] for i in range(row)]


if __name__ == '__main__':
    mat = create_mat(4, 4)
    print_clockwise(mat)
