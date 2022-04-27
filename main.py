# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import data_loader


def spam_detector():
    data = data_loader.load_data()
    # print(type(data))
    print(f'Shape of data frame is {data.shape}')

    print(data.head())
    # print(data.tail())

    print(data.CATEGORY.value_counts())


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    spam_detector()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
