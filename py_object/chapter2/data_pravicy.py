class SecretString:
    # python 无强制变量保护，但是可以通过在变量前设置下划线来提示
    # 也阔以称为 命名改装
    """A not-at-all secure way to store a secret string."""
    def __init__(self, plain_string, pass_phrase):
        self.__plain_string = plain_string
        self.__pass_phrase = pass_phrase

    def decrypt(self, pass_phrase):
        if pass_phrase == self.__pass_phrase:
            return self.__plain_string
        else:
            return ''


if __name__ == '__main__':
    test = SecretString("secret", "password")
    print(test.decrypt("password"))

    # 使用类名加属性名称可以访问命名改装的属性
    print(test._SecretString__plain_string)