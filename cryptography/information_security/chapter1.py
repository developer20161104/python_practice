from typing import List


# 厄拉多塞师筛法
def prime_collection(n: int) -> List[int]:
    if n == 1:
        return []

    # 设置筛选列表以及因子筛选长度
    arr, x = [1] * (n + 1), int(n ** 0.5)
    for i in range(2, x + 1):
        # 直接使用切片进行倍数删除
        arr[2 * i::i] = [0] * len(arr[2 * i::i])

    # 需要删除前两个值0,1
    return [index for index, x in enumerate(arr) if x][2:]


if __name__ == '__main__':
    print(prime_collection(200))
