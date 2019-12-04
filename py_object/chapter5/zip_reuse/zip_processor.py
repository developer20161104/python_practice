import os
import shutil
import zipfile
from pathlib import Path


# 父类提供公共方法，需要与子类共同使用
class ZipProcessor:
    def __init__(self, zipname):
        self.zipname = zipname
        self.temp_directory = Path("unzipped-{}".format(zipname[:-4]))

    def process_zip(self):
        self.unzip_files()
        # 此方法由子类实现
        self.process_files()
        self.zip_files()

    def unzip_files(self):
        self.temp_directory.mkdir()

        with zipfile.ZipFile(self.zipname) as zips:
            zips.extractall(str(self.temp_directory))

    def zip_files(self):
        with zipfile.ZipFile(self.zipname, 'w') as file:
            for filename in self.temp_directory.iterdir():
                file.write(str(filename), filename.name)

        shutil.rmtree(str(self.temp_directory))