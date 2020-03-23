import time


def log_call(func):
    # 在函数中添加的操作
    def wrap(*args, **kwargs):
        now = time.time()

        print('calling {0} with {1} and {2}'.format(
            func.__name__, args, kwargs
        ))

        return_value = func(*args, **kwargs)
        print('executed {0} in {1}ms'.format(
            func.__name__, time.time() - now
        ))

        return return_value

    return wrap


# 使用装饰器语法也能实现操作
# 但是此类语法仅适用于自定义装饰器
@ log_call
def test1(a,b,c):
    print('\ttest1 called')


def test2(a,b):
    print('\ttest2 called')


def test3(a,b):
    print('\ttest3 called')
    time.sleep(1)


if __name__ == '__main__':
    # test1 = log_call(test1)
    test2 = log_call(test2)
    test3 = log_call(test3)

    test1(1,2,3)
    test2(4,b=5)
    test3(6,7)

