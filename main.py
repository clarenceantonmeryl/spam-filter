# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import data_loader
import data_visualizer
import data_processor


def spam_detector():
    data = data_loader.load_data()
    # print(type(data))
    print(f'Shape of data frame is {data.shape}')

    # print(data.head())
    # print(data.tail())

    # data_visualiser.draw_pie_chart(data)


    msg = "All work and no play makes Jack a dull boy. To be or not to be. ??? Nobody expects the Spanish Inquisition!"
    words = data_processor.tokenize_message(msg)
    filtered_words = data_processor.remove_stopwords(words)
    print(filtered_words)

    data_processor.download_resource()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    spam_detector()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
