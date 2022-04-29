import numpy as np

import utility.constant as constant

import os
from os.path import join

import pandas as pd

HAM_1_PATH = 'data/source/spam-assassin-corpus/ham-1'
HAM_2_PATH = 'data/source/spam-assassin-corpus/ham-2'
SPAM_1_PATH = 'data/source/spam-assassin-corpus/spam-1'
SPAM_2_PATH = 'data/source/spam-assassin-corpus/spam-1'

# DATA_SOURCE_JSON_FILE = 'data/source/data-source.json'
DATA_SOURCE_JSON_FILE = 'data/source/data-source-original.json'
DATA_STEMMED_CSV_FILE = 'data/source/data-stemmed.csv'
VOCABULARY_CSV_FILE = 'data/source/vocabulary.csv'

TRAINING_CSV_FILE = 'data/train/train.csv'
TESTING_CSV_FILE = 'data/train/test.csv'

TRAINING_TXT_FILE = 'data/train/train.txt'
TESTING_TXT_FILE = 'data/train/test.txt'


def extract_email(file) -> str:

    """
    Extract the email body from the file path.

    :param file: Relative file path.
    :return: Email body without the email header part.
    """

    with open(file=file, mode='r', encoding='latin-1') as message:
        lines = message.readlines()

    body: str = ""

    try:
        body_index = lines.index('\n')

    except ValueError:
        pass

    else:
        body_lines = lines[body_index:]

        for line in body_lines:
            if line == '\n':
                body_lines.remove(line)

        body = '\n'.join(line.strip() for line in body_lines if line != '\n')

    finally:
        return body


def email_body_generator(path):

    """
    Email body generator function
    :param path: Path of a directory where raw emails are present.
    :return: Yields the extracted email body.
    """

    for root, dirctory_names, file_names in os.walk(path):
        for file_name in file_names:
            file_path = join(root, file_name)

            body = extract_email(file=file_path)
            yield file_name, body


def get_dataframe_from_path(path, category) -> pd.DataFrame:

    """
    Get the DataFrame object of all emails in a path.

    :param path: Path of the directory of raw emails.
    :param category: Category of the emails.
    :return: DataFrame of the all emails in the supplied directory path.
    """

    rows = []
    row_names = []
    for file_name, body in email_body_generator(path=path):
        rows.append({'MESSAGE': body, 'CATEGORY': category})
        row_names.append(file_name)

    return pd.DataFrame(rows, index=row_names)


def get_data_source_from_raw_files() -> pd.DataFrame:

    """
    Fetch the DataFrame of all source emails.

    :return: A DataFrame of all emails from 'Spam Assassin Corpus'
    """

    # Load data from spam 1 path
    spam_emails = get_dataframe_from_path(
        path=SPAM_1_PATH,
        category=constant.SPAM_CATEGORY
    )

    # Append data from spam 2 path
    spam_emails = pd.concat(
        [
            spam_emails,
            get_dataframe_from_path(path=SPAM_2_PATH, category=constant.SPAM_CATEGORY)
        ]
    )

    # print(spam_emails.head())
    # print(spam_emails.shape)

    # Load data from ham 1 path
    ham_emails = get_dataframe_from_path(
        path=HAM_1_PATH,
        category=constant.HAM_CATEGORY
    )

    # Append data from ham 2 path
    ham_emails = pd.concat(
        [
            ham_emails,
            get_dataframe_from_path(path=HAM_2_PATH, category=constant.HAM_CATEGORY)
        ]
    )

    # print(ham_emails.head())
    # print(ham_emails.shape)

    data = pd.concat([spam_emails, ham_emails])

    # print(data.shape)
    # print(data.head())
    # print(data.tail())

    # Check null
    # print(data['MESSAGE'].isnull().values.any())
    # print(data[data.MESSAGE.isnull()].index)
    # print(data.index.get_loc('.DS_Store'))
    #
    # print(data[692:695])

    # print(data['MESSAGE'].isnull().values.any())
    # print(data[data.MESSAGE.isnull()].index)
    # print(data[692:695])

    # Check empty
    # print((data.MESSAGE.str.len() == 0).any())

    # Locate empty
    # print(data(data.MESSAGE.str.len() == 0).index)
    # data.index.get_loc('.DS_Store')

    try:
        # Remove System File Entries from Dataframe
        data = data.drop(['cmds', 'DS_Store'])
        # data = data.drop(['.DS_Store'])
        # data.drop(['cmds', 'DS_Store'], inplace=True)
    except KeyError:
        print("System files not found.")

    # Set DOC_IDs
    document_ids = range(0, len(data.index))
    data['DOC_ID'] = document_ids
    data['FILE_NAME'] = data.index
    data.set_index('DOC_ID', inplace=True)

    # print(data.head())
    # print(data.tail())

    return data


def save_json_data(data: pd.DataFrame) -> None:

    """
    Save the DataFrame as a JSON file.

    :param data: The DataFrame to be saved as JSON file.
    """

    data.to_json(DATA_SOURCE_JSON_FILE)


def load_data_from_json() -> pd.DataFrame:

    """
    Load and return a DataFrame from JSON file.

    :return: A DataFrame from the JSON file.
    """

    data = pd.read_json(DATA_SOURCE_JSON_FILE)

    # Set DOC_IDs
    document_ids = range(0, len(data.index))
    data['DOC_ID'] = document_ids
    data.set_index('DOC_ID', inplace=True)

    return data


def load_dataframe() -> pd.DataFrame:

    """
    Load the source data as DataFrame.

    :return: A DataFrame of the data source.
    """

    print(f"Loading data frame from: {DATA_SOURCE_JSON_FILE}")

    # data = get_data_source_from_raw_files()

    # save_json_data(data)

    data = load_data_from_json()

    # print(data.shape)

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

    # print("Shape", data[data.CATEGORY == 1].shape)

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

    return data


def save_stemmed_csv_data(data: pd.DataFrame):

    """
    Save the DataFrame as a JSON file.

    :param data: The DataFrame to be saved as CSV file.
    """

    data.to_csv(DATA_STEMMED_CSV_FILE)


def load_stemmed_data() -> pd.DataFrame:

    """
    Load the stemmed data as DataFrame.

    :return: A DataFrame of the stemmed data.
    """

    print(f"Loading stemmed data frame from: {DATA_STEMMED_CSV_FILE}")

    data = pd.read_csv(DATA_STEMMED_CSV_FILE)
    # Set DOC_IDs
    data.set_index('DOC_ID', inplace=True)

    return data


def save_csv_vocabulary(vocabulary: pd.DataFrame):
    """
    Save the DataFrame as a JSON file.

    :param vocabulary: The DataFrame to be saved as JSON file.
    """

    # vocabulary.to_csv(VOCABULARY_CSV_FILE, index_label=vocabulary.index.name, header=vocabulary.VOCAB_WORD.name)
    vocabulary.to_csv(VOCABULARY_CSV_FILE)


def load_vocabulary() -> pd.DataFrame:

    """
    Load the vocabulary data as DataFrame.

    :return: A DataFrame of the data source.
    """

    print(f"Loading vocabulary data frame from: {VOCABULARY_CSV_FILE}")

    vocabulary = pd.read_csv(VOCABULARY_CSV_FILE)
    # Set WORD_IDs
    vocabulary.set_index('WORD_ID', inplace=True)

    return vocabulary


def save_training_data(training_data: pd.DataFrame):
    training_data.to_csv(TRAINING_CSV_FILE)
    np.savetxt(TRAINING_TXT_FILE, training_data, fmt='%d')


def save_testing_data(testing_data: pd.DataFrame):
    testing_data.to_csv(TESTING_CSV_FILE)
    np.savetxt(TESTING_TXT_FILE, testing_data, fmt='%d')


def load_training_dataframe() -> pd.DataFrame:
    training_data = pd.read_csv(TRAINING_CSV_FILE)
    return training_data


def load_testing_dataframe() -> pd.DataFrame:
    testing_data = pd.read_csv(TESTING_CSV_FILE)
    return testing_data


def load_sparse_data() -> (np.ndarray, np.ndarray):
    sparse_train_data = np.loadtxt(TRAINING_TXT_FILE, delimiter=' ', dtype=int)
    # print(type(sparse_train_data))
    # print(sparse_train_data[:10])
    print("Rows in Training", sparse_train_data.shape)

    sparse_test_data = np.loadtxt(TESTING_TXT_FILE, delimiter=' ', dtype=int)
    # print(type(sparse_test_data))
    # print(sparse_test_data[:10])
    print("Rows in Training", sparse_test_data.shape)

    print("Emails in training", np.unique(sparse_train_data[:, 0]).size)
    print("Emails in testing", np.unique(sparse_test_data[:, 0]).size)

    return sparse_train_data, sparse_test_data


def make_full_matrix(sparse_matrix, doc_id_index=0, word_id_index=1, category_index=2, occurrence_index=3):

    """
    Form a full matrix from a sparse matrix. Return a pandas dataframe.

    :param sparse_matrix: numpy array
    :param doc_id_index: position of the document id in the sparse matrix. Default: 1st column
    :param word_id_index: position of the word id in the sparse matrix. Default: 2nd column
    :param category_index: position of the label (spam is 1, nonspam is 0). Default: 3rd column
    :param occurrence_index: position of occurrence of word in sparse matrix. Default: 4th column
    :return: Return a pandas dataframe of full matrix
    """

    column_names = ['DOC_ID'] + ['CATEGORY'] + list(range(0, constant.VOCABULARY_SIZE))
    index_names = np.unique(sparse_matrix[:, 0])

    full_matrix = pd.DataFrame(index=index_names, columns=column_names).fillna(value=0)

    for row in range(sparse_matrix.shape[0]):
        doc_id = sparse_matrix[row][doc_id_index]
        word_id = sparse_matrix[row][word_id_index]
        category = sparse_matrix[row][category_index]
        occurrence = sparse_matrix[row][occurrence_index]

        full_matrix.at[doc_id, 'DOC_ID'] = doc_id
        full_matrix.at[doc_id, 'CATEGORY'] = category
        full_matrix.at[doc_id, word_id] = occurrence

    full_matrix.set_index('DOC_ID', inplace=True)
    return full_matrix


def progress():
    sparse_train_data, sparse_test_data = load_sparse_data()

    full_train_data = make_full_matrix(sparse_train_data)
    print(full_train_data)

