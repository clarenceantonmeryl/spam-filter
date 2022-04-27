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

    # print(data.at[2, "MESSAGE"])

    # data_visualizer.draw_pie_chart(data)

    # data_processor.download_resource()

    # print(data_processor.get_stopwords())

    sample_text = "a b c d e f g h i j k l m n o p q r s t u v w x y z " \
                  "All work and no <strong>play makes</strong> Jack a dull boy. To be or not to be. " \
                  "??? Nobody expects the Spanish Inquisition!"
    no_html = data_processor.remove_html_tags(sample_text)
    words = data_processor.tokenize_text(no_html)
    filtered_words = data_processor.remove_stopwords(words)
    print(filtered_words)

    print(data_processor.clean_stemmed_tokens(sample_text))
    print(data_processor.clean_stemmed_tokens(data.at[2, "MESSAGE"], is_stem=True))

    # print(data_processor.remove_html_tags(data.at[2, "MESSAGE"]))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    spam_detector()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
