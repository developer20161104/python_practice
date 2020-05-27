# 使用 pytest 来进行测试时，使用 py.test filename 来编译即可
# python 函数即对象，因此可以直接进行测试
def test_int_float():
    assert 1 == 1.0


# 以类来进行测试
class TestNumbers:
    def test_int_float(self):
        assert 1 == 1.0

    def test_int_str(self):
        assert 1 == '1'
