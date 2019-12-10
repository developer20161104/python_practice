import time


class Color:
    def __init__(self, rgb_value, name):
        # 使用下划线表示私有变量
        self._name = name
        self.rgb_value = rgb_value

    def _set_name(self, name):
        if not name:
            raise Exception("Invalid name")
        self._name = name

    def _get_name(self):
        return self._name

    # 创建一个名为name的属性，并让其拥有两个方法,使得变量访问受限
    # 虽然还是能够直接访问_name变量，但是此时已经不是编程人员的事情了
    name = property(_get_name, _set_name)


class Silly:
    # 看起来还有点迷
    # 使用装饰器来设置getter setter与deleter 方法
    @property
    def silly(self):
        """This is a silly property"""
        print("you are getting silly")
        # 设置时效
        return self._silly if time.localtime(time.time() - self._old_time ).tm_sec < 3 else ""

    @silly.setter
    def silly(self, value):
        print("you are making silly {}".format(value))
        self._silly = value
        # 从创建开始进行计时
        self._old_time = time.time()

    @silly.deleter
    def silly(self):
        print("kill silly")
        del self._silly


class AverageList(list):
    # 自定义取值方法，创建属性
    @ property
    def average(self):
        return sum(self) / len(self)


if __name__ == '__main__':
    c = Color("#0000ff", "bright red")

    # 访问属性
    print(c.name)

    c.name = "hello"
    print(c.name)
    # 此时设置变量时也会进行验证
    # c.name = ""

    s = Silly()
    s.silly = "test"
    # 添加休眠时间来手动增加延时
    # time.sleep(4)
    print(s.silly)

    del s.silly

    lists = AverageList([1, 2, 3, 4])
    # 使用装饰器来增加类中的属性
    print(lists.average)
