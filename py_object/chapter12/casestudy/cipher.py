# 通过测试来进行驱动的开发，确实能使代码更加强健
class VigenereCipher:
    def __init__(self, keyword:str):
        self.keyword = keyword

    def encode(self, plaintext:str):
        # return 'XECWQXUIVCRKHWA'
        cipher = []
        # 过滤空格符与大小写带来的影响
        plaintext = plaintext.replace(' ', '').upper()
        keyword = self.extend_keyword(len(plaintext))

        # 对于两个列表同时提取元素
        for k, p in zip(plaintext, keyword):
            cipher.append(combine_characer(p, k))

        return ''.join(cipher)
        # 使用str()转化列表还是有问题

    def extend_keyword(self, lens: int) -> str:
        repeats = lens // len(self.keyword) + 1
        # 这个写法有点意思，前面负责重复，后面限制长度就行了
        return (self.keyword * repeats)[:lens]


def combine_characer(plain: str, keyword: str):
    plain_num = ord(plain.upper()) - ord('A')
    keyword_num = ord(keyword.upper()) - ord('A')

    # 核心算法：求和取模
    return chr(ord('A') + (plain_num + keyword_num) % 26)
