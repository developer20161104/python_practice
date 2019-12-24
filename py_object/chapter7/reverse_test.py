normal_list = [1, 2, 3, 4, 5]


class CustomSequence():
    def __len__(self):
        return 5

    # 迭代过程每次都会调用该方法
    def __getitem__(self, index):
        return "x{0}".format(index + 1)


class FunkyBackwards():
    # 更改了reverse方法会使得原有的reversed方法失效
    def __reversed__(self):
        return "BACKWARDS!"


if __name__ == '__main__':
    for seq in normal_list, CustomSequence(), FunkyBackwards():
        print("\n{}: ".format(seq.__class__.__name__), end="")

        for item in reversed(seq):
            # end属性将在每一个尾部添加结束符
            print(item, end=", ")
