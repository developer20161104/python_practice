from cipher import *


# 测试硬编码
def test_encoding():
    cipher = VigenereCipher("TRAIN")
    encoded = cipher.encode('ENCODEDINPYTHON')
    assert encoded == 'XECWQXUIVCRKHWA'


# 测试单个转化
def test_encode_character():
    cipher = VigenereCipher('TRAIN')
    encoded = cipher.encode('E')
    assert encoded == 'X'


# 测试小写与空格符
def test_encode_lowercase():
    cipher = VigenereCipher('TRain')
    encoded = cipher.encode('encoded in Python')
    assert encoded == 'XECWQXUIVCRKHWA'


# 测试函数
def test_combine_character():
    assert combine_characer('E', 'T') == 'X'
    assert combine_characer('n', 'R') == 'E'


# 测试内部方法
def test_extend_keyword():
    cipher = VigenereCipher('TRAIN')
    extended = cipher.extend_keyword(11)
    assert extended == 'TRAINTRAINT'
