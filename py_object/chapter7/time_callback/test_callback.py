from time_func import Timer
import datetime


def format_time(message, *args):
    # 输出时间格式化
    now = datetime.datetime.now().strftime("%I:%M:%S")
    print(message.format(*args, now=now))


def one(timer):
    format_time("{now}: called one")


def two(timer):
    format_time("{now} called two")


def three(timer):
    format_time("{now} called three")


class Repeator:
    def __init__(self):
        self.count = 0

    # 重复调用自身
    def repeater(self, timer):
        format_time("{now}: repeat {0}", self.count)

        self.count += 1
        timer.call_after(5, self.repeater)


class Repeator2:
    def __init__(self):
        self.count = 0

    # 通过内建函数来调用自身
    def __call__(self, timer):
        format_time("{now}: repeat {0}", self.count)

        self.count += 1
        timer.call_after(5, self)


if __name__ == '__main__':
    timer = Timer()
    # 通过计时器回调函数来选择调用顺序
    timer.call_after(1, one)
    timer.call_after(2, one)
    timer.call_after(2, two)
    timer.call_after(4, two)
    timer.call_after(3, three)
    timer.call_after(6, three)

    repeater = Repeator()

    # 重复执行
    """
    timer.call_after(5, repeater.repeater)
    format_time("{now}: starting")

    timer.run()
    """

    # 使用内建call来执行
    timer.call_after(5, Repeator2())
    format_time("{now}: starting")

    timer.run()
