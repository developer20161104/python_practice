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

