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


if __name__ == '__main__':
    # 由于方法采用列表式参数，类似指针，因此对于接收的参数数量没有限制，但是没有分别设置参数名
    get_pages()
    get_pages("http://helloworld.com")
    get_pages("http://helloworld.com", "http://localhost:8080")

    options = Option(username="hello", password="world", debug=True)
    print(options.options, '\n', options['port'])
