from collections import defaultdict
from email_send import send_email


class MailingList:
    """manage groups of e-mail addresses for sending e-mails"""

    # 使用每个邮箱去匹配组
    def __init__(self):
        self.email_map = defaultdict(set)

    def add_to_group(self, email, group):
        self.email_map[email].add(group)

    def emails_in_group(self, *groups):
        # 收集一个或多个组中所有邮箱地址
        return set(e for e, g in self.email_map.items() if g & set(groups))

    def send_emailing(self, subject, message, from_addr, *groups, header=None):
        emails = self.emails_in_group(*groups)

        send_email(subject, message, from_addr, *emails, headers=header)


if __name__ == '__main__':
    # 看得有点迷
    m = MailingList()

    m.add_to_group("friend1@123.com", "friend")
    m.add_to_group("friend2@123.com", "friend")
    m.add_to_group("family1@hello.com", "family")
    m.add_to_group("prof@test.com", "professional")

    m.send_emailing("A party", "Friend and family only: a party", "friend", "family", "professional",  header={"Reply-To": "me2@qq.com"})
