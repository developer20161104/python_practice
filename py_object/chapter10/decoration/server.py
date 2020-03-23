import socket
import gzip
from io import BytesIO


class LogSocket:
    # 打印日志的装饰器
    # 在发送前打印相关日志
    def __init__(self, socket):
        self.socket = socket

    def send(self, data):
        # 多出来这一步
        print('Send {0} to {1}'.format(
            data, self.socket.getpeername()[0]))

        self.socket.send(data)

    def close(self):
        self.socket.close()

    # def getpeername(self):
    #     self.socket.getpeername()


class GzipSocket:
    # 压缩数据并发送的装饰器
    def __init__(self, socket):
        self.socket = socket

    def send(self, data):
        buf = BytesIO()
        # 先压缩再发送
        with gzip.GzipFile(fileobj=buf, mode='w') as zipfile:
            zipfile.write(data)

        self.socket.send(buf.getvalue())

    def close(self):
        self.socket.close()


def respond(client):
    response = input('enter a value')

    # 实质是调用LogSocket类中重写的方法
    client.send(bytes(response, 'utf8'))
    client.close()


if __name__ == '__main__':
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('127.0.0.1', 2401))
    server.listen(1)
    # 模拟配置变量
    log_send = 1
    compress_hosts = ['localhost']

    try:
        while True:
            client, addr = server.accept()
            # respond(LogSocket(client))
            # 添加多个装饰器
            if log_send:
                client = LogSocket(client)
            # if client.getpeername()[0] in compress_hosts:
            client = GzipSocket(client)

            respond(client)

    #  需要用异常来进行结束，因此不能用with
    finally:
        server.close()