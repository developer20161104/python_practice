import re
import sys
import json
from pathlib import Path


# 匹配诸如 /** operate name **/ 格式
DIRECTIVE_RE = re.compile(
    r'/\*\*\s*(include|loopover|variable|endloop|loopvar)'
    r'\s*([^ *]*)\s*\*\*/')


class TemplateEngine:
    def __init__(self, infilename, outfilename, contextfilename):
        # 文件内容
        self.template = open(infilename).read()
        # ？
        self.working_dir = Path(infilename).absolute().parent
        # 指定当前字符所在位置
        self.pos = 0
        # 输出文件
        self.outfile = open(outfilename, 'w')
        # ?
        with open(contextfilename) as contextfile:
            self.context = json.load(contextfile)

    def process(self):
        # 指定关键字参数
        match = DIRECTIVE_RE.match(self.template, pos=self.pos)

        while match:
            # 先将之前的文件逐一写入
            self.outfile.write(self.template[self.pos:match.start()])
            # # 更新起始位
            # self.pos = match.end()

            # ??
            try:
                directive, argument = match.groups()
                method_name = 'process_{}'.format(directive)
                getattr(self, method_name)(match, argument)
            except:
                pass

            match = DIRECTIVE_RE.match(self.template, pos=self.pos)

        # 写入后续字符
        self.outfile.write(self.template[self.pos:])

    # 查找引入的文件并插入文件内容
    def process_include(self, match, argument):
        with (self.working_dir / argument).open() as includefile:
            self.outfile.write(includefile.read())
            self.pos = match.end()

    # 从上下文字典中查找变量名，找不到使用默认字符串
    def process_variable(self, match, argument):
        self.outfile.write(self.context.get(argument, ''))
        self.pos = match.end()

    def process_loopover(self, match, argument):
        # 列表下标
        self.loop_index = 0
        # 从上下文字典中获取的列表
        self.loop_list = self.context.get(argument, [])
        # 指示遍历完成后跳转的位置
        self.pos = self.loop_pos = match.end()

    def process_loopvar(self, match, argument):
        self.outfile.write(self.loop_list[self.loop_index])
        self.pos = match.end()

    def process_endloop(self, match, argument):
        self.loop_index += 1
        if self.loop_index >= len(self.loop_list):
            self.pos = match.end()

            del self.loop_list
            del self.loop_index
            del self.loop_pos
        else:
            self.pos = self.loop_pos


if __name__ == '__main__':
    infilename, outfilename, contextfilename = sys.argv[1:]

    # 处理模板
    engine = TemplateEngine(infilename, outfilename, contextfilename)
    engine.process()