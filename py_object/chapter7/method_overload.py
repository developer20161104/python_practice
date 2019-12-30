# 列表式：不限制数量
def get_pages(*links):
    for link in links:
        print(link)


class Option:
    default_options = {
        'port': 21,
        'host': 'localhost',
        'username': None,
        'password': None,
        'debug': False,
    }

    # 字典式参数
    def __init__(self, **kwargs):
        # 利用字典中的参数进行自动更新
        self.options = dict(Option.default_options)
        # 注意此处不能进行简单合并，毕竟需要更新的是self中的字典
        self.options.update(kwargs)

    def __getitem__(self, key):
        # 能使用索引语法进行读取
        return self.options[key]


def show_args(arg1, arg2, arg3="THREE"):
    print(arg1, arg2, arg3)


if __name__ == '__main__':
    # 由于方法采用列表式参数，类似指针，因此对于接收的参数数量没有限制，但是没有分别设置参数名
    get_pages()
    get_pages("http://helloworld.com")
    get_pages("http://helloworld.com", "http://localhost:8080")

    options = Option(username="hello", password="world", debug=True)
    print(options.options, '\n', options['port'])

    # 参数解包的应用
    some_args = range(3)

    more_args = {
        "arg1": "ONE",
        "arg2": "TWO"
    }

    print("Unpacking sequence")
    # 将自定义参数代入方法并根据其内容自行作为列表参数
    show_args(*some_args)

    print("Unpacking dict")
    # 自行作为字典参数
    show_args(**more_args)