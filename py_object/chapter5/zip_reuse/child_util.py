from zip_processor import ZipProcessor
import sys
import os


class ZipReplace(ZipProcessor):
    def __init__(self, filename, search_string, replace_string):
        super().__init__(filename)

        self.search_string = search_string
        self.replace_string = replace_string

    # 子类实现具体功能
    def process_files(self):
        for filename in self.temp_directory.iterdir():
            with filename.open() as file:
                contents = file.read()

            contents = contents.replace(self.search_string, self.replace_string)

            with filename.open("w") as file:
                file.write(contents)


if __name__ == '__main__':
    # 实现并调用父类的委托方法
    # 极大提高了代码的可读性
    ZipReplace(*sys.argv[1:4]).process_zip()