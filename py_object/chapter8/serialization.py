import pickle
import json


class Contact:
    def __init__(self, first, last):
        self.first = first
        self.last = last

    @property
    def full_name(self):
        return '{} {}'.format(self.first, self.last)


class ContactEncoder(json.JSONEncoder):
    def default(self, obj):
        # 自定义编码器
        # 判断类是否为指定类型
        if isinstance(obj, Contact):
            return {
                'is_contact': True,
                'first': obj.first,
                'last': obj.last,
                'full': obj.full_name
            }
        # 如果不是指定类型，则有父类方法来操作
        # 但是有bug：自定义类型还是无法识别？？
        return super().default(obj)


def decode_contact(dic):
    if dic.get('is_contact'):
        return Contact(dic['first'], dic['last'])
    return dic


if __name__ == '__main__':
    some_data = ['a list', 'containing', 5, 'value including another list',
                 ['inner', 'list']]

    # 存储的位置为文件
    # 进行序列化存储
    with open('pickled_list', 'wb') as file:
        pickle.dump(some_data, file)

    # 读取序列化数据
    with open('pickled_list', 'rb') as file:
        load_file = pickle.load(file)

    print(load_file)
    assert load_file == some_data

    # 存储为bytes类型
    data = pickle.dumps('hello world', protocol=4)

    # 提取
    print(data, '\n', pickle.loads(data))

    # json化类
    # 如果只需要数据，则对其dict属性进行序列化即可
    c = Contact('hello', 'world')
    print(json.dumps(c.__dict__))

    # 通过自定义编码器来获取属性值
    # print(json.dumps(c, cls=ContactEncoder))

    data = json.dumps(c, cls=ContactEncoder)

    # 通过loads指定加载函数来将json结构体转化为python对象
    print(json.loads(data, object_hook=decode_contact).full_name)

    