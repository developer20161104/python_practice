import sys
import shutil
import zipfile
from pathlib import Path


# 用来管理其他对象的管理员对象
class ZipReplace:
    def __init__(self, filename, search_string, replace_string):
        self.filename = filename
        self.search_string = search_string
        self.replace_string = replace_string
        self.temp_directory = Path("unzipped-{}".format(filename))

    # 委托方法
    def zip_find_replace(self):
        self.unzip_files()
        self.find_replace()
        self.zip_files()

    # 解压当前目录下文件
    def unzip_files(self):
        self.temp_directory.mkdir()
        with zipfile.ZipFile(self.filename) as zips:
            zips.extractall(str(self.temp_directory))

    # 替换文件内容
    def find_replace(self):
        for filename in self.temp_directory.iterdir():
            with filename.open() as file:
                contents = file.read()

            contents = contents.replace(self.search_string, self.replace_string)

            with filename.open("w") as file:
                file.write(contents)

    # 压缩文件
    def zip_files(self):
        with zipfile.ZipFile(self.filename, 'w') as file:
            for filename in self.temp_directory.iterdir():
                file.write(str(filename), filename.name)
            shutil.rmtree(str(self.temp_directory))


# 通过参数读取进行压缩文件内容的替换，可作为API调用批量处理文件
if __name__ == '__main__':
    ZipReplace(*sys.argv[1:4]).zip_find_replace()
