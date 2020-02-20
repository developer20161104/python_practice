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

    # 格式化输出2号
    #
    orders = [('burger', 2, 5),
              ('fries', 3.5, 1),
              ('cola', 1.75, 3)]

    print("PRODUCT    QUANTITY    PRICE    SUBTOTAL")
    for product, price, quantity in orders:
        subtotal = price * quantity
        # {0:10s}第一个变量是占据10个字符的字符串
        # {1:^9d}第二个变量是占据9字符并且 居中 的数字，其余字符以空格输出
        # {2: <8.2f}第三个变量按照长度为8保留两位小数左对齐
        # {3: >7.2f}第四个变量按照长度为7保留两位小数右对齐
        print("{0:10s}{1: ^9d}    ${2: <8.2f}${3: >7.2f}".format(product, quantity, price, subtotal))

    # 字节转化为文本
    # 使用bytes类中的.decode方法来进行解码

    characters = b'\x63\x6c\x69\x63\x68\xe9'
    print(characters)
    # 输出解码
    print(characters.decode("latin"))

    # 将文本转化为字节
    print("cliché".encode("latin"))

    # 构造可变字节字符串
    b = bytearray(b'abcdesf')
    b[4:6] = b'\x15\xa3'
    print(b)
