import unittest
from typing import List
import sys


# 通过继承并实现相关方法来完成单元测试
class CheckNumbers(unittest.TestCase):

    # 初始化方法，适用于对相同输入不同方法的测试
    def setUp(self) -> None:
        pass

    def test_init_float(self):
        self.assertEqual(1, 1.0)

    # def test_wrong(self):
    #     self.assertEqual(1,'1')

    # 给定用例来进行测试
    def test_zero(self):
        # self.assertRaises(ZeroDivisionError,
        #                   average,
        #                   [])

        # 另一种写法，使用with
        with self.assertRaises(ZeroDivisionError):
            average([])

    # 使用装饰器来过滤错误
    @ unittest.expectedFailure
    def test_fails(self):
        self.assertEqual(False, True)

    @ unittest.skip("Test is useless")
    def test_skip(self):
        self.assertEqual(False, True)

    # 根据python版本确定是否运行
    @ unittest.skipIf(sys.version_info.minor == 7,
                      "broken on 3.6")
    def test_skipif(self):
        self.assertEqual(False, True)

    # 根据系统来确定
    @ unittest.skipUnless(sys.platform.startswith('linux'),
                          "broken unless on linux")
    def test_skipunless(self):
        self.assertEqual(False, True)


def average(seq: List[int]):
    return sum(seq) / len(seq)


if __name__ == '__main__':
    # 模块化的单元测试，每个测试都只针对代码中的最小单元
    unittest.main()