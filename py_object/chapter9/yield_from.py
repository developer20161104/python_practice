# 使用 yield from 的原因
# 相比于直接使用yield实现生成器，
# yield from在对内部可迭代元素自动进行yield的基础上
# 还进行了一系列的异常捕捉判断


# 子生成器
def average_gen():
    total = 0
    count = 0
    average = 0

    while True:
        # 应用了协程
        new_num = yield average
        if not new_num:
            break
        count += 1
        total += new_num
        average = total / count

    # 结束当前协程
    return total, count, average


# 委托生成器
# 建立一个双向通道，用于调用方与子生成器的交互
def proxy_gen():
    while True:
        # 将生成器嵌套到 yield from 中
        yield from average_gen()


# 调用方
def main():
    calc_average = proxy_gen()
    next(calc_average)  # 预激活生成器

    print(calc_average.send(10))  # 传入值
    print(calc_average.send(20))  #
    print(calc_average.send(30))  #

    # 如果没有yield from 的内部异常捕捉，此处会报错
    calc_average.send(None)

    # 如果后续继续调用，则会开启一个新的协程


if __name__ == '__main__':
    main()
