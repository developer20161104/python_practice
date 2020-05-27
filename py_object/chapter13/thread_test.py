from threading import Thread


class InputReader(Thread):
    def run(self) -> None:
        self.line_of_text = input()


if __name__ == '__main__':
    print("Enter some text and press enter:")
    thread = InputReader()
    # 子线程用于接收键盘键入
    # 以并发模式执行
    # thread.start()
    # 对于非并发模式，则不会进入下面的循环
    # thread.run()

    count = result = 1
    # 主线程计算统计数据
    while thread.is_alive():
        result = count * count
        count += 1

    print('calculated squares up to {0} * {0} = {1}'.format(
        count, result
    ))

    print('while you typed {}'.format(thread.line_of_text))
