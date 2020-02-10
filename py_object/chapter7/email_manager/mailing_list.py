from collections import defaultdict
from email_send import send_email


class MailingList:
    """manage groups of e-mail addresses for sending e-mails"""

    # 使用每个邮箱去匹配组
    def __init__(self, data_file):
        # 设置一个以set作为value的字典
        self.email_map = defaultdict(set)

        # save
        self.data_file = data_file

    # enter 与 exit 内置方法作用与with上下文管理器的开始与结束
    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()

    def add_to_group(self, email, group):
        self.email_map[email].add(group)

    # 将组员放到一起，但是在此处好像又是所有接收方邮件是在同一个列表里
    # 唯一的功能是过滤不存在与组列表中的其它接收方邮件信息，思想还是蛮妙的
    def emails_in_group(self, *groups):
        # 收集一个或多个组中所有邮箱地址
        return set(e for e, g in self.email_map.items() if g & set(groups))

    def send_emailing(self, subject, message, from_addr, *groups, header=None):
        emails = self.emails_in_group(*groups)
        # 嵌套封装
        send_email(subject, message, from_addr, *emails, headers=header)

    # 将数据保存到磁盘中
    def save(self):
        with open(self.data_file, 'w') as file:
            for email, groups in self.email_map.items():
                file.write(
                    '{} {}\n'.format(email, ''.join(groups))
                )

    def load(self):
        self.email_map = defaultdict(set)

        try:
            with open(self.data_file) as file:
                for line in file:
                    email, groups = line.strip().split()
                    # 将单个字符串转化为set集合适配先前的字典
                    groups = set(groups.split(','))
                    self.email_map[email] = groups
        except IOError:
            print('file not exit')


if __name__ == '__main__':

    m = MailingList('emails.txt')

    # m.add_to_group("friend1@123.com", "friend")
    # m.add_to_group("friend2@123.com", "friend")
    # m.add_to_group("family1@hello.com", "family")
    # m.add_to_group("family2@hello.com", "family")
    # m.add_to_group("prof@test.com", "professional")

    # m.save()
    m.send_emailing("A party", "Friend and family only: a party", "friend", "family", "professional",  header={"Reply-To": "me2@qq.com"})

    # m.load()

    # 使用with进行直接调用
    with MailingList('emails.txt') as test:
        print(test.email_map)
