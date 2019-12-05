# 将文档模拟为一个字符数组
class Document:
    def __init__(self):
        self.characters = []
        self.cursor = Cursor(self)
        self.filename = ''

    # 插入指定光标位置，注意需要调用的是内部的cursor而非外部的对象
    def insert(self, character):
        # 判断当前是否为字符串，字符串需要进行转化后操作
        if not hasattr(character, 'character'):
            character = Character(character)
        self.characters.insert(self.cursor.position, character)
        self.cursor.forward()

    # 删除当前光标指向的字符
    def delete(self):
        del self.characters[self.cursor.position]

    # 保存文件到指定位置
    def save(self):
        with open(self.filename, 'w') as f:
            f.write(''.join(self.characters))

    @ property
    def string(self):
        return "".join([str(c) for c in self.characters])


class Cursor:
    def __init__(self, document):
        self.document = document
        self.position = 0

    def forward(self):
        self.position += 1

    def back(self):
        self.position -= 1

    def home(self):
        # self.position = 0
        try:
            # 此处会产生异常
            while self.document.characters[self.position - 1] != '\n':
                self.position -= 1
                if self.position == 0:
                    break
        except IndexError as _:
            print("当前为空！")

    def end(self):
        self.position = -1


# 逐字符包装实现富文本，由于不需要多字符操作，因此也不用继承str类
class Character:
    def __init__(self, character, bold=False, italic=False, underline=False):
        self.character = character
        self.bold = bold
        self.italic = italic
        self.underline = underline

    # str 修改用户界面的显示，将实例字符串化
    def __str__(self):
        bold = "*" if self.bold else ""
        italic = "/" if self.italic else ""
        underline = "_" if self.underline else ""

        # 结果返回
        return bold + italic + underline + self.character


if __name__ == '__main__':
    d = Document()
    d.cursor.home()
    d.insert(Character('h', bold=True))
    # 添加此种时会经过字符串向Character的转化
    d.insert('e')
    d.insert('l')
    d.insert('l')
    d.insert(Character('o', italic=True))
    d.insert('\t')
    d.insert('a')
    d.cursor.home()
    d.insert(Character('F', underline=True))
    print(d.string)