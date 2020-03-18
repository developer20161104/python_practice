import re


# 通过给定正则式来匹配
def match_regex(filename, regex):
    with open(filename) as file:
        lines = file.readlines()

        for line in reversed(lines):
            match = re.match(regex, line)

            if match:
                # 协程
                regex = yield match.groups()[0]


# 遍历第一个任务的结果，并给出正则式
def get_details(filename):
    ERROR_RE = 'XFS ERROR (\[sd[a-z]\])'
    matcher = match_regex(filename, ERROR_RE)

    # 先获取sd号码
    # 在取得结果后会对当前环境进行保存
    device = next(matcher)
    while True:
        # 根据sd来找出总线标识符
        bus = matcher.send('(sd \S+) {}.*'.format(re.escape(device)))
        # 由标识符来找到最终结果，并取消对反括号的匹配
        serial = matcher.send('{} \(SERIAL=([^)]*)\)'.format(bus))
        # 生成器
        yield serial

        # 继续向后匹配（妙啊）
        # 由于yield会保存返回前的环境，因此会向后继续匹配
        device = matcher.send(ERROR_RE)


if __name__ == '__main__':
    # 很像linux中的管道符
    for serial_number in get_details('EXAMPLE_LOG.log'):
        print(serial_number)