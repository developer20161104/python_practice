def tally():
    score = 0

    while True:
        # 使用协程
        increment = yield score
        score += increment


if __name__ == '__main__':
    test_1 = tally()
    # 初始化,即先将方法执行到内部的循环
    # 在yield处等候下一个传参
    next(test_1)
    # 传入值
    print(test_1.send(10))
    print(test_1.send(30))

    test_2 = tally()
    next(test_2)
    print(test_2.send(5))
    print(test_2.send(15))

