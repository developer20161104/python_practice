def get_Merge(*args):
    # for things in args:
    #     yield from things

    # 等价写法，不过可以在过程中添加过滤器
    for x in args:
        yield from (i for i in x)


if __name__ == '__main__':
    res = []
    for i in get_Merge([], [1,2,3], ['a','b'],[1],['a']):
        res.append(i)

    print(res)