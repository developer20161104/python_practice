class OneOnly:
    _singleton = None

    def __new__(cls, *args, **kwargs):
        # 如果还没有创建，则进行创建，否则直接返回
        if not cls._singleton:
            cls._singleton = super(OneOnly, cls
                                   ).__new__(cls, *args, **kwargs)

        return cls._singleton


if __name__ == '__main__':
    n1 = OneOnly()
    n2 = OneOnly()

    print(n1)
    print(n1 == n2)