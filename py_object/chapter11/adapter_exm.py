import datetime


# 原始功能类
# 只能接受特定形式的字符串，因此需要使用适配器模式
class AgeCalculator:
    def __init__(self, birthday: str):
        # 切割字符串
        self.year, self.month, self.day = (int(x) for x in birthday.split('-'))

    def calculate_age(self, date: str):
        year, month, day = (
            int(x) for x in date.split('-')
        )

        age = year - self.year
        # python 的层级比较
        if (month, day) < (self.month, self.day):
            age -= 1

        return age


# 将datetime.date datetime.time对象转化为特定的string类型
class DateAdapter:
    def _str_date(self, dates: datetime.date):
        return dates.strftime('%Y-%m-%d')

    def __init__(self, birthday: datetime.date):
        # 先转化为字符串
        birthday = self._str_date(birthday)
        # 再调用原始方法
        self.calculator = AgeCalculator(birthday)

    # 注意此处的对象为datetime.date 或者 datetime.time类型
    def get_age(self, date):
        date = self._str_date(date)
        # 每次调用前进行初始化，转化为可识别字符串
        return self.calculator.calculate_age(date)


if __name__ == '__main__':
    # 使用计算年龄的适配器
    bd = DateAdapter(datetime.date(1997, 10, 20))
    print(bd.get_age(datetime.datetime.now()))
