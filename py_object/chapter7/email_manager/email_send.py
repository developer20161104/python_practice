import smtplib
from email.mime.text import MIMEText


def send_email(subject, message, from_addr, *to_addr, host="localhost", port=1025, headers=None):
    # 设置headers默认值为空
    headers = {} if headers is None else headers
    # 信息构建
    email = MIMEText(message)
    email['Subject'] = subject
    email['From'] = from_addr

    for header, value in headers.items():
        email[header] = value

    # 邮件发送
    sender = smtplib.SMTP(host, port)
    for addr in to_addr:
        del email['To']
        email['To'] = addr

        sender.sendmail(from_addr, addr, email.as_string())


if __name__ == '__main__':
    send_email("A test subject", "The massage content", "from@123.email", "to1@1234.com", "hello@qq.com")
