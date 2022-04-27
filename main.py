# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd

import data_loader
import data_processor
# import data_visualizer
import data_wordcloud

import time


def spam_detector():

    # data_processor.download_resource()

    data = data_loader.load_data()
    # print(type(data))
    print(f'Shape of data frame is {data.shape}')

    # print(data.head())
    # print(data.tail())

    # print(data.at[2, "MESSAGE"])
    # print(data.iat[2, 0])
    # print(data.iat[2, 1])
    # print(data.iat[2, 2])

    # print(data.iloc[3:9])

    # emails_series = data.MESSAGE.iloc[0:3]
    # print(type(emails_series))
    # print(emails_series)


    """

    doc_ids_spam = data[data.CATEGORY == 1].index
    doc_ids_ham = data[data.CATEGORY == 0].index

    # nested_list_all = data.MESSAGE.apply(data_processor.clean_stemmed_tokens)
    nested_list_all = data.MESSAGE.apply(data_processor.clean_tokens)
    nested_list_spam = nested_list_all.loc[doc_ids_spam]
    nested_list_ham = nested_list_all.loc[doc_ids_ham]

    print(nested_list_spam.shape)
    print(nested_list_ham.shape)

    flat_list_spam = [word for sub_list in nested_list_spam for word in sub_list]
    flat_list_ham = [word for sub_list in nested_list_ham for word in sub_list]
    print("Spam Word Count:", len(flat_list_spam))
    print("Ham Word Count: ", len(flat_list_ham))

    words_spam = pd.Series(flat_list_spam).value_counts()
    words_ham = pd.Series(flat_list_ham).value_counts()

    print(words_ham.shape[0])
    print(words_spam.shape[0])

    print("Top Spam Words:\n", words_spam[:10])
    print("Top Ham Words:\n", words_ham[:10])

    
    """

    start_time = time.time()

    data_wordcloud.generate_email_wordcloud(data)



    # data_wordcloud.sample_machine_learning()


    # print("Shape", data[data.CATEGORY == 1].shape)

    # spam_message_series = data[data.CATEGORY == 1].MESSAGE.apply(data_processor.clean_stemmed_tokens)
    # spam_word_list = [word for sub_list in spam_message_series for word in sub_list]

    # print(len(spam_word_list))

    end_time = time.time()

    print("Time Taken: ", end_time - start_time)

    # data_visualizer.draw_pie_chart(data)
    # print(data_processor.clean_stemmed_tokens(data.at[2, "MESSAGE"]))
    # print(data_processor.remove_html_tags(data.at[2, "MESSAGE"]))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    spam_detector()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
