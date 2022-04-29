# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd

import data_loader
import data_pre_processor
# import data_visualizer
import data_processor
import data_wordcloud

import time


def spam_detector():


    # data_pre_processor.download_resource()

    # data = data_loader.load_data()

    # data_visualizer.draw_pie_chart(data)
    # data_wordcloud.generate_email_wordcloud(data)

    # data = data_processor.generate_stemmed_messages(data)
    # data_loader.save_stemmed_csv_data(data)

    data = data_loader.load_stemmed_data()

    print(data)

    # vocabulary = data_processor.generate_vocabulary(data)
    # data_loader.save_csv_vocabulary(vocabulary)

    # print("Vocabulary 1:\n", vocabulary)

    vocabulary = data_loader.load_vocabulary()
    vocabulary_set = set(vocabulary.VOCAB_WORD)
    vocabulary_index = data_processor.generate_vocabulary_index(vocabulary)
    print("Vocabulary:\n", vocabulary)
    print('app' in vocabulary_set)
    print('email location in vocab:', vocabulary_index.get_loc('email'))

    # nested_list_all = data.MESSAGE
    # clean_email_lengths = [len(str(message).split(' ')) for message in nested_list_all.values]
    # print(max(clean_email_lengths))
    # print(clean_email_lengths.index(7671))
    # print(np.argmax(clean_email_lengths))
    # print(nested_list_all[5404])
    # print(data_pre_processor.tokenize_text(data.at[5404, 'MESSAGE']))

    '''
    x_train, x_test, y_train, y_test = data_processor.generate_train_test_data(data)
    '''

    start_time = time.time()

    '''
    sparse_training = data_processor.make_sparse_matrix(
        x_dataframe=x_train,
        vocabulary_index=vocabulary_index,
        y_dataframe=y_train)

    print("sparse_train_df\n", sparse_training)
    print("SHAPE sparse_train_df\n", sparse_training.shape)

    training_data = sparse_training.groupby(['DOC_ID', 'WORD_ID', 'LABEL']).sum().reset_index()

    print("TRAINING 1:")
    print(training_data)

    data_loader.save_csv_training_data(training_data=training_data)

    '''

    training_data = data_loader.load_training_data()

    print("TRAINING:")
    print(training_data)


    '''
    sparse_testing = data_processor.make_sparse_matrix(
        x_dataframe=x_test,
        vocabulary_index=vocabulary_index,
        y_dataframe=y_test)

    print("sparse_train_df\n", sparse_testing)
    print("SHAPE sparse_train_df\n", sparse_testing.shape)

    testing_data = sparse_testing.groupby(['DOC_ID', 'WORD_ID', 'LABEL']).sum().reset_index()

    print("TESTING 1:")
    print(testing_data)

    data_loader.save_csv_testing_data(testing_data=testing_data)

    '''

    testing_data = data_loader.load_testing_data()

    print("TESTING:")
    print(testing_data)


    '''
    print(f'Shape of data frame is {data.shape}')

    doc_ids_spam = data[data.CATEGORY == 1].index
    doc_ids_ham = data[data.CATEGORY == 0].index

    # nested_list_all = data.MESSAGE.apply(data_pre_processor.clean_stemmed_tokens)
    nested_list_all = data.MESSAGE.apply(data_pre_processor.clean_msg_no_html)
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

    '''

    end_time = time.time()

    print("Time Taken: ", end_time - start_time)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    spam_detector()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
