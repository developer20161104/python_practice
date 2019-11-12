# 继承了内部类List，并为其添加方法
class ContactList(list):
    def search(self, name):
        """return all contacts that contain the search value in their name
        """
        match_lists = []
        for contact in self:
            if name in contact.name:
                match_lists.append(contact)

        return match_lists


class Contact:
    all_contacts = ContactList()

    def __init__(self, name, email):
        self.name = name
        self.email = email
        Contact.all_contacts.append(self)


# 继承了联系人类，具备其构造器
class Supplier(Contact):
    def order(self, order):
        # 类似c的写法，在特定位置为其赋值
        print("send '{}' to '{}'".format(order, self.name))


# 改写超类方法，super会实例化超类
class Friend(Contact):
    def __init__(self, name, email, phone):
        super().__init__(name, email)
        self.phone = phone


if __name__ == '__main__':
    # 测试基本的继承
    test = Supplier("hello", "qq@hello.com")
    test.order("show methods")
