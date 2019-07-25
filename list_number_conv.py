def number2list(number):
    n = 10
    listofzeros = [0] * n
    listofzeros[number] = 1
    return listofzeros


def list2number(_list):
    return _list.index(1)

if __name__ == '__main__':
    number = 6
    print(number2list(number))
    test = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    print(list2number(test))