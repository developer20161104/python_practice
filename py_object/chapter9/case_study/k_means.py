import csv
from random import random
import math
from collections import Counter

dataset_filename = 'colors.csv'


# 将csv文件内容进行格式转换
def load_colors(filename):
    with open(filename) as file:
        lines = csv.reader(file)

        for line in lines:
            # 转换为浮点数加颜色名元组
            yield tuple(float(y) for y in line[:3]), line[3]


# 构造随机颜色
def generate_colors(count=100):
    for i in range(count):
        yield (random(), random(), random())


# 计算两点之间的距离
def color_distance(color_1, color_2):
    channels = zip(color_1, color_2)
    sum_distance = 0

    for c1, c2 in channels:
        sum_distance += (c1-c2)**2

    return math.sqrt(sum_distance)


# k_means算法
def nearest_neighbors(model_colors, num_neighbors):
    model = list(model_colors)

    target = yield
    while True:
        # 返回数据结构(distance, (r,g,b), name)
        distance = sorted(
            ((color_distance(c[0], target), c) for c in model),
        )

        # 找到最邻近的前k个值，并返回(r,g,b)
        target = yield [
            d[1] for d in distance[:num_neighbors]
        ]


# 维护一个文件，利用协程输出结果
def write_result(filename='output.csv'):
    with open(filename, 'w') as file:

        writer = csv.writer(file)
        while True:
            color, name = yield
            writer.writerow(list(color) + [name])


# 求取最佳猜测
def name_colors(get_neighbors):
    color = yield
    while True:
        near = get_neighbors.send(color)
        # 选取最接近的一个
        name_guess = Counter(
            n[1] for n in near
        ).most_common(1)[0][0]

        # 返回最佳猜测，并接受新的颜色
        color = yield name_guess


# 最终整合
def process_colors(dataset_filename='colors.csv'):
    model_colors = load_colors(dataset_filename)

    # 创建各类协程与生成器
    get_neighbors = nearest_neighbors(model_colors, 5)
    get_color_name = name_colors(get_neighbors)
    output = write_result()

    # 初始化
    next(output)
    next(get_neighbors)
    next(get_color_name)

    for color in generate_colors():
        # 通过随机一个(r,g,b)来预测最接近的颜色名称
        name = get_color_name.send(color)
        # 将结果送入输出文件
        output.send((color, name))


if __name__ == '__main__':
    process_colors()