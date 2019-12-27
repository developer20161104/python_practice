import random
import string


class StringJoiner(list):
    def __enter__(self):
        # 开始之前执行此方法
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # with 会在脱离其块后执行此方法
        self.result = "".join(self)


if __name__ == '__main__':
    with StringJoiner() as lists:
        for i in range(15):
            lists.append(random.choice(string.ascii_letters))

    # 通过改写exit内部方法来对执行完with后的部分进行处理
    print(lists, "\n", lists.result)
