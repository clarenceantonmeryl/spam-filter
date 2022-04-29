import pandas as pd

import data_pre_processor
from utility import constant

from sklearn.model_selection import train_test_split


def generate_stemmed_messages(data: pd.DataFrame):
    data['MESSAGE'] = data.MESSAGE.apply(data_pre_processor.clean_stemmed_text)
    return data


def generate_vocabulary(data: pd.DataFrame, is_stem=True):

    if not is_stem:
        nested_list_all = data.MESSAGE.apply(data_pre_processor.clean_stemmed_tokens)
    else:
        nested_list_all = data.MESSAGE.apply(data_pre_processor.tokenize_text)

    flat_list_all = [word for sub_list in nested_list_all for word in sub_list]

    unique_words = pd.Series(flat_list_all).value_counts()
    frequent_words = unique_words[0:constant.VOCABULARY_SIZE]

    print(frequent_words)
    print(frequent_words[1])

    vocabulary = pd.DataFrame(
        {
            'VOCAB_WORD': frequent_words.index.values,
            'WORD_COUNT': frequent_words.values
        },
        index=list(range(0, constant.VOCABULARY_SIZE))
    )
    vocabulary.index.name = "WORD_ID"

    print(vocabulary)

    return vocabulary


def generate_vocabulary_index(vocabulary: pd.DataFrame):
    return pd.Index(vocabulary.VOCAB_WORD)


# Generate Features & a Sparse Matrix

def generate_train_test_data(data: pd.DataFrame, is_stem=True):

    print('generate_features'.upper())

    if not is_stem:
        nested_list_all = data.MESSAGE.apply(data_pre_processor.clean_stemmed_tokens)
    else:
        nested_list_all = data.MESSAGE.apply(data_pre_processor.tokenize_text)

    # print(type(nested_list_all))
    # print(nested_list_all)

    word_columns_df = pd.DataFrame.from_records(nested_list_all.tolist())
    word_columns_df.index.name = "DOC_ID"

    # print(word_columns_df.head())
    # print(word_columns_df.shape)

    x_train, x_test, y_train, y_test = train_test_split(word_columns_df, data.CATEGORY, test_size=0.3, random_state=42)

    print(f'No. of training sample: {x_train.shape}')
    print(f'Fraction of training set: {x_train.shape[0] / word_columns_df.shape[0]}')
    print("X_train\n", x_train.head())
    print("y_train\n", y_train.head())

    print("X_test\n", x_test.head())
    print("y_test\n", y_test.head())

    return x_train, x_test, y_train, y_test


def make_sparse_matrix(x_dataframe: pd.DataFrame, vocabulary_index, y_dataframe):

    """
    Returns sparse matrix as DataFrame

    :param x_dataframe: A DataFrame with words in the columns with a document id as index (x_train or x_test)
    :param vocabulary_index: index of words ordered by word_id
    :param y_dataframe: Category of a series (y_train or y_test)
    :return:
    """

    rows = x_dataframe.shape[0]
    columns = x_dataframe.shape[1]

    word_set = set(vocabulary_index)

    sparse_matrix = []

    for row in range(rows):
        for column in range(columns):

            word = x_dataframe.iat[row, column]

            if word in word_set:
                doc_id = y_dataframe.index[row]
                word_id = vocabulary_index.get_loc(word)
                category = y_dataframe.at[doc_id]

                item = {
                    'DOC_ID': doc_id,
                    'WORD_ID': word_id,
                    'LABEL': category,
                    'OCCURRENCE': 1
                }

                sparse_matrix.append(item)

    return pd.DataFrame(sparse_matrix)


def extract_excluded_doc_ids(data: pd.DataFrame, training_data: pd.DataFrame, testing_data: pd.DataFrame) -> list:

    # IDs from source data
    data_doc_ids = set(data.index.values.tolist())

    # IDs from training and testing data
    training_doc_ids = set(training_data.DOC_ID)
    testing_doc_ids = set(testing_data.DOC_ID)

    # Combine training and testing IDs
    included_ids = training_doc_ids.copy()
    included_ids.update(testing_doc_ids)

    # Sort the excluded IDs
    excluded_ids = sorted(data_doc_ids - included_ids)

    return excluded_ids
