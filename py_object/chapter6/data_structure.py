import datetime
from collections import namedtuple


def middle(stock, date):
    symbol, current, high, low = stock
    return (high + low) // 2, date


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
    print(dicts)
