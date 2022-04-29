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

TRAINING_DATA_FILE = 'data/train/train.csv'
TESTING_DATA_FILE = 'data/train/test.csv'


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


def load_data() -> pd.DataFrame:

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


def save_csv_training_data(training_data: pd.DataFrame):
    training_data.to_csv(TRAINING_DATA_FILE)


def load_training_data() -> pd.DataFrame:
    training_data = pd.read_csv(TRAINING_DATA_FILE)
    return training_data


def save_csv_testing_data(testing_data: pd.DataFrame):
    testing_data.to_csv(TESTING_DATA_FILE)


def load_testing_data() -> pd.DataFrame:
    testing_data = pd.read_csv(TESTING_DATA_FILE)
    return testing_data
