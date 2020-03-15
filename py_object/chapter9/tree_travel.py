class File:
    def __init__(self, name):
        self.name = name


def walk(file):
    if isinstance(file, Folder):
        # 为文件添加目录
        yield file.name + '/'
        for f in file.child:
            # 迭代目录
            yield from walk(f)
    else:
        # 不是目录的话返回文件名
        yield file.name


# 添加子目录
class Folder(File):
    def __init__(self,name):
        super().__init__(name)
        self.child = []


if __name__ == '__main__':
    root = Folder('')
    etc = Folder('etc')
    root.child.append(etc)
    # etc目录下文件
    root.child.append(File('password'))
    root.child.append(File('groups'))

    # etc子集目录及内含文件
    httpd = Folder('httpd')
    etc.child.append(httpd)
    httpd.child.append(File('httpd.conf'))

    var = Folder('var')
    root.child.append(var)
    # var下目录
    log = Folder('log')
    var.child.append(log)
    log.child.append(File('message'))
    log.child.append(File('kernel'))

    #           基本架构
    #             root
    #      etc            var
    # httpd   twofile   log
    # httpd.conf       twofile

    for i in walk(root):
        print(i)
