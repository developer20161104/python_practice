# 输出数据结构类型
class Node:
    def __init__(self, tag_name, parent=None):
        self.parent = parent
        self.tag_name = tag_name

        self.children = []
        self.text = ''

    def __str__(self):

        if self.text:
            return self.tag_name + ':' + self.text
        else:
            return self.tag_name


# 解析器
class Parser:
    def __init__(self, parse_string):
        # 所要解析的内容
        self.parse_string = parse_string
        # 树形结构的顶端结点
        self.root = None
        # 当前处理的结点
        self.current_node = None

        self.state = FirstTag()

    def process(self, remaining_string):
        remaining = self.state.process(remaining_string, self)
        if remaining:
            self.process(remaining)

    def start(self):
        self.process(self.parse_string)


# 初始状态
class FirstTag:
    def process(self, remaining_string, parser):
        i_start_tag = remaining_string.find('<')
        i_end_tag = remaining_string.find('>')

        # 获取标签
        tag_name = remaining_string[i_start_tag+1:i_end_tag]
        root = Node(tag_name)

        # 状态初始化
        parser.root = parser.current_node = root
        parser.state = ChildNode()

        return remaining_string[i_end_tag+1:]


class ChildNode:
    def process(self, remaining_string, parser):
        stripped = remaining_string.strip()

        # 进行状态的切换
        if stripped.startswith('</'):
            parser.state = CloseTag()
        elif stripped.startswith('<'):
            parser.state = OpenTag()
        else:
            parser.state = TextNode()

        return stripped


# 将新创建的节点添加到原来的当前节点的child中
class OpenTag:
    def process(self, remaining_string, parser):
        i_start_tag = remaining_string.find('<')
        i_end_tag = remaining_string.find('>')
        tag_name = remaining_string[i_start_tag+1:i_end_tag]

        # 主要操作
        # 新建一个节点
        node = Node(tag_name, parser.current_node)
        # 将其作为子节点并向下递归
        parser.current_node.children.append(node)
        parser.current_node = node
        parser.state = ChildNode()

        return remaining_string[i_end_tag+1:]


class CloseTag:
    def process(self, remaining_string, parser):
        i_start_tag = remaining_string.find('<')
        i_end_tag = remaining_string.find('>')
        assert remaining_string[i_start_tag+1] == '/'

        # 判断标签是否统一
        tag_name = remaining_string[i_start_tag+2:i_end_tag]
        assert tag_name == parser.current_node.tag_name

        # 返回父节点
        parser.current_node = parser.current_node.parent
        parser.state = ChildNode()

        return remaining_string[i_end_tag+1:].strip()


# 提取结束标签之前的文本内容
class TextNode:
    def process(self, remaining_string, parser):
        i_start_tag = remaining_string.find('<')
        text = remaining_string[:i_start_tag]

        parser.current_node.text = text
        parser.state = ChildNode()

        return remaining_string[i_start_tag:]


if __name__ == '__main__':
    with open('simple_example.xml') as file:
        contents = file.read()
        p = Parser(contents)
        p.start()

        nodes = [p.root]
        while nodes:
            node = nodes.pop()
            print(node)

            nodes = node.children + nodes
