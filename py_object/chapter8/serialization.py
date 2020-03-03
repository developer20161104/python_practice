import pickle

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