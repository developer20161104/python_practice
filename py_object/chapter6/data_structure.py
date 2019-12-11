import datetime
import string
from collections import namedtuple
from collections import defaultdict
num_item = 0


def middle(stock, date):
    symbol, current, high, low = stock
    return (high + low) // 2, date


def tuple_counter():
    global num_item
    num_item += 1
    return num_item, []


def test_dict(dicts: dict, key: str, val: int):
    dicts[key] = val
    return dicts


if __name__ == '__main__':

    # 元组解包：将多值返回使用方法转化
    mid_value, date = middle(("Fb", 8, 7, 9), datetime.date(2014, 10, 30))
    print(mid_value, date)

    # 命名元组的使用
    # 首先需要设置名称以及各属性（以空格分开）
    Stock = namedtuple("Stock", "symbol current high low")

    stock = Stock("FB", 12, high=9, low=7)
    # 注意的是，元组不能进行修改
    print(stock, stock.current)

    # python 会为自动为字典添加已包含或并未出现的键值对：不需要判断了！
    dicts = {"a": 1}
    # dicts["b"] = 5
    test_dict(dicts, "hakwfh", 12)
    # 直接调用的话会报错，使用defaultdict就不会有这种情况
    # print(dicts["hello"])

    # defaultdict 可以通过函数调用完成自定义设置
    test = defaultdict(tuple_counter)
    # 在返回的列表位置添加元素
    test["a"][1].append("hello")
    test["a"][1].append("test")
    test["b"][1].append("world")
    print(test)

    # 直接迭代输出所有的字母（包含大小写）  妙啊
    print(list(string.ascii_letters)+[" "])

