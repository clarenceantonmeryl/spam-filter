# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import data_source_loader as data_loader


def spam_detector():
    data = data_loader.get_data()
    print(type(data))
    print(data.shape)

    print(data.head())
    print(data.tail())


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    spam_detector()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
