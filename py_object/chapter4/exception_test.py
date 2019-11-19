# 重写Exception类的初始化方法，并添加新方法
class InvalidWithdrawal(Exception):
    def __init__(self, balance, amount):
        super().__init__("account does not have ${} ".format(amount))
        self.balance = balance
        self.amount = amount

    def overage(self):
        return self.amount - self.balance


if __name__ == '__main__':
    try:
        raise InvalidWithdrawal(25, 50)
    except InvalidWithdrawal as e:
        print("your withdrawal is more than your balance by ${}".format(e.overage()))
    finally:
        print("execute code in any circumstances")
