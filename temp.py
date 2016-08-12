# coding:utf-8

class father(object):
    def __init__(self):
        self.name = 'Father'

    def __call__(self):
        print('father call')
        self.build()
        print(self.name)

    def build(self):
        print('father build')


class son(father):
    def __init__(self):
        super(son, self).__init__()
        self.name = 'Son'

    def call(self):
        print('son call')

    def build(self):
        print('son build')


if __name__ == '__main__':
    # f = father()
    # print(f.name)
    # s = son()
    # print(s.name)
    # s()
    l = [1, 2, 3]
    print(l[-1])