from typing import List


class CapitalIterable:
    def __init__(self, string):
        self.string = string

    # 对于可迭代对象，需要实现迭代器实例
    def __iter__(self):
        return CapitalIterator(self.string)


class CapitalIterator:
    def __init__(self, string):
        self.words = [w.capitalize() for w in string.split()]
        self.index = 0

    # 实现next方法来进行迭代
    def __next__(self):
        if self.index == len(self.words):
            raise StopIteration

        word = self.words[self.index]
        self.index += 1
        return word

    # 对于迭代器只需要实现自身实例即可
    def __iter__(self):
        return self


def filter_self(string: List[str]):
    for cur_str in string:
        if 'a' in cur_str:
            yield cur_str
            # 多个yield生成器时，顺序执行
            yield cur_str + ' test_two'


def filter_two(str_sort: List[str]):
    # 将可迭代对象作为生成器的输入
    # 内部元素会逐一迭代
    yield from str_sort


if __name__ == '__main__':
    # 可迭代对象会把字符串分割为子串，并将首字母大写
    cap = CapitalIterable('hello world the python object')
    # 将可迭代对象送入iter方法进行迭代
    it = iter(cap)

    # while True:
    #     try:
    #         print(next(it))
    #     except StopIteration:
    #         break

    # 使用for也能达到同样效果
    # for i in it:
    #     print(i)

    string = 'hello a the dagger apply this is a test'
    # 调用yield生成器
    # for s in filter_self(string):
    #     print(s)

    # 注意区分添加了from的区别
    for s in filter_two(string.split()):
        print(s)
