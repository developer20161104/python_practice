class Inventory:
    def __init__(self):
        self.observers = []
        self._product = None
        self._quantity = 0

    def attach(self, observer):
        self.observers.append(observer)

    @ property
    def product(self):
        return self._product

    @ product.setter
    def product(self, value):
        self._product = value
        # 更新
        self._update_observer()

    @ property
    def quantity(self):
        return self._quantity

    @ quantity.setter
    def quantity(self, value):
        self._quantity = value
        # 更新
        self._update_observer()

    def _update_observer(self):
        # 通知各个观察者 ‘；’【】；】、
        for ob in self.observers:
            ob()


# 一个简单的观察者对象
class ConsoleObserver:
    def __init__(self, inventory):
        self.inventory = inventory

    # 直接调用观察者对象必须实现__call__方法
    def __call__(self, *args, **kwargs):
        print(self.inventory.product)
        print(self.inventory.quantity)


if __name__ == '__main__':
    i = Inventory()

    # 观察核心类的变化
    c1 = ConsoleObserver(i)
    c2 = ConsoleObserver(i)

    # 核心类添加观察者
    i.attach(c1)
    i.attach(c2)

    # 核心类调用setter方法时，会通知所添加的各个观察者
    i.product='widget'
    i.quantity = 5