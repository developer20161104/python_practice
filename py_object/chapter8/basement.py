if __name__ == '__main__':
    # 区别于split，partition只切割一次字符串
    strs = 'ww.baidu.com'

    print(strs.partition('.'))

    # join方法连接字符串神器
    t = ['test', 'connect', 'method']
    print('@'.join(t))

    # 使用format格式化匹配字符串{}，并且能在其中加入索引下标指定匹配元素
    # 一旦使用数字索引，则全部位置都需要
    # 也能使用关键字参数（类似C语言体系）
    template = "hello {0} is the {1} so that {0}"
    print(template.format('first', 'second'))

    # 需要跳过格式化时，使用双括号
    # example: Java格式化
    template_2 = """
    public class {0} {{
         public static void main(String[] args) {{
        System.out.print("{1}");
        }}
    }}"""

    print(template_2.format("MyClass", "print('hello world')"))

    # 容器查询测试
    emails = ('a@example.com', 'b@example.com')

    message = {
        'subject': "just a test",
        'message': 'here is an email'
    }

    # 模板测试时注意赋值的书写方式
    template_3 = """
    From: <{0[0]}>
    To: <{0[1]}>
    Subject: {message[subject]}
    ·{message[message]}"""

    # 一个默认参数，一个指定参数
    print(template_3.format(emails, message=message))

    # 格式化输出
    subtotal = 12.32
    tax = 0.07 * subtotal
    total = tax + subtotal

    # 通过下标指定以及格式化输出以及指定参数名输出
    # 0.2f表示如果小于1时，小数点左边含0，并保留小数点后两位输出浮点数
    print("Sub: ${0:0.2f} Tax: ${1:0.2f} Total: ${total:0.2f}".format(subtotal, tax, total=total))