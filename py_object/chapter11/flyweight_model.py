import weakref


class CarModel:
    _models = weakref.WeakValueDictionary()

    # 生产者用于创建实例
    def __new__(cls, model_name, *args, **kwargs):
        model = cls._models.get(model_name)

        if not model:
            model = super().__new__(cls)
            cls._models[model_name] = model

        return model

    # 对实例进行初始化
    def __init__(self, model_name, air=False, tilt=False,
                 cruise_control=False, power_locks=False,
                 alloy_wheels=False, usb_charger=False):

        # 只有在第一次出现的时候会进行初始化
        if not hasattr(self, 'initted'):
            self.model_name = model_name
            self.air = air
            self.tilt = tilt
            self.cruise_control = cruise_control
            self.power_locks = power_locks
            self.alloy_wheels = alloy_wheels
            self.usb_charger = usb_charger
            self.initted = True

    def check_serial(self, serial_number):
        print('Sorry, we are unable to check the serial number'
              '{0} on the {1} at this time'.format(serial_number, self.model_name))


# 定义一个新类来存储额外的信息，以及对享元对象的引用
class Car:
    def __init__(self, model, color, serial):
        self.model = model
        self.color = color
        self.serial = serial

    def check_serial(self):
        return self.model.check_serial(self.serial)


if __name__ == '__main__':
    dx = CarModel('Fix dx')
    lx = CarModel('Fix lx', air=True, cruise_control=True,
                  power_locks=True, tilt=True)

    car1 = Car(dx, 'blue', '12345')
    car2 = Car(dx, 'black', '12347')
    car3 = Car(lx, 'red', '12348')

    print(id(lx))
    # 删除并且强制进行垃圾回收
    del lx
    del car3
    import gc

    gc.collect()

    lx = CarModel('new lx', tilt=True)
    print(id(lx))

    # 由于之前已经创建过对象，因此此对象不会重新创建
    lx = CarModel('new lx')
    print(lx.tilt)
