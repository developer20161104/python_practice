import re
import sys


def match_str(patterns, search_string):
    # patterns = sys.argv[1]
    # search_string = sys.argv[2]

    match = re.match(patterns, search_string)

    # 内部变量还能这么用？？
    if match:
        template = "'{}' matches pattern '{}'"
    else:
        template = "'{}' does not matches pattern '{}'"

    print(template.format(search_string, patterns))


if __name__ == '__main__':
    search_thing = 'hello world'

    pattern = 'hello world'

    # 字符串的完全匹配
    if re.match(search_thing, pattern):
        print('regex matches')

    # . 匹配任意字符
    # yes
    match_str('hel.o', 'heloo')

    # [ch]匹配ch中的任意字符
    # yes
    match_str('hel[lp]o', 'helpo')

    # [^ch]匹配非ch字符
    # no
    match_str('hel[^lp]', 'helpo')

    # [a-z]匹配任意小写字母，后面的表示数量
    # yes
    match_str('[a-z]{5}', 'hello')

    # 使用反斜杠表示转义\，更推荐在字符串前面使用r表示去掉字符串中字符的特殊属性
    # yes
    match_str(r'0.[0-9]{2}', '0.98')

    # \w 表示所有数字下划线与字母
    # yes
    match_str('\w{3}', 'a_1')

    # * 表示任意，+ 表示一或多， ？表零或一个
    # yes
    match_str('[A-Z][a-z]* [a-z]*', 'A string')
