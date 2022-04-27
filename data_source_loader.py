import utility.constant as constant

import os
from os.path import join

import pandas as pd

HAM_1_PATH = 'data/source/spam-assassin-corpus/ham-1'
HAM_2_PATH = 'data/source/spam-assassin-corpus/ham-2'
SPAM_1_PATH = 'data/source/spam-assassin-corpus/spam-1'
SPAM_2_PATH = 'data/source/spam-assassin-corpus/spam-1'

DATA_SOURCE_JSON_FILE = 'data/source/data-source.json'


def extract_email(file):
    with open(file=file, mode='r', encoding='latin-1') as message:
        lines = message.readlines()

    body = None

    try:
        body_index = lines.index('\n')

    except ValueError:
        pass

    else:
        body = lines[body_index:]

        for line in body:
            if line == '\n':
                body.remove(line)

        body = '\n'.join(line.strip() for line in body if line != '\n')

    finally:
        return body


def email_body_generator(path):
    for root, dirctory_names, file_names in os.walk(path):
        for file_name in file_names:
            file_path = join(root, file_name)

            body = extract_email(file=file_path)
            yield file_name, body


def dataframe_from_path(path, category):
    rows = []
    row_names = []
    for file_name, body in email_body_generator(path=path):
        rows.append({'MESSAGE': body, 'CATEGORY': category})
        row_names.append(file_name)

    return pd.DataFrame(rows, index=row_names)


def get_data_source_from_raw_files():
    spam_emails = dataframe_from_path(
        path=SPAM_1_PATH,
        category=constant.SPAM_CATEGORY
    )

    spam_emails = pd.concat(
        [
            spam_emails,
            dataframe_from_path(path=SPAM_2_PATH, category=constant.SPAM_CATEGORY)
        ]
    )

    # print(spam_emails.head())
    # print(spam_emails.shape)

    ham_emails = dataframe_from_path(
        path=HAM_1_PATH,
        category=constant.HAM_CATEGORY
    )

    ham_emails = pd.concat(
        [
            ham_emails,
            dataframe_from_path(path=HAM_2_PATH, category=constant.HAM_CATEGORY)
        ]
    )

    # print(ham_emails.head())
    # print(ham_emails.shape)

    df = pd.concat([spam_emails, ham_emails])

    # print(df.shape)
    # print(df.head())
    # print(df.tail())

    # Check null
    # print(df['MESSAGE'].isnull().values.any())
    # print(df[df.MESSAGE.isnull()].index)
    # print(df.index.get_loc('.DS_Store'))
    #
    # print(df[692:695])

    # df = df.drop(['.DS_Store'])
    # print(df['MESSAGE'].isnull().values.any())
    # print(df[df.MESSAGE.isnull()].index)
    # print(df[692:695])

    # Check empty
    # print((df.MESSAGE.str.len() == 0).any())

    # Locate empty
    # print(df(df.MESSAGE.str.len() == 0).index)
    # df.index.get_loc('.DS_Store')

    # Remove System File Entries from Dataframe
    # df = df.drop(['cmds', 'DS_Store'])
    # df.drop(['cmds', 'DS_Store'], inplace=True)

    # Add Document IDs to Track Emails in Dataset
    document_ids = range(0, len(df.index))
    df['DOC_ID'] = document_ids
    df['FILE_NAME'] = df.index
    df.set_index('DOC_ID', inplace=True)

    print(df.head())
    print(df.tail())

    return df


def save_data(df):
    df.to_json(DATA_SOURCE_JSON_FILE)


def get_data_from_json():
    df = pd.read_json(DATA_SOURCE_JSON_FILE)

    # Add Document IDs to Track Emails in Dataset
    document_ids = range(0, len(df.index))
    df['DOC_ID'] = document_ids
    df['FILE_NAME'] = df.index
    df.set_index('DOC_ID', inplace=True)

    return df


def get_data():
    print(f"Hello world {constant.SPAM_CATEGORY} {DATA_SOURCE_JSON_FILE}")
    # data = get_data_source_from_raw_files()
    # save_data(data)
    data = get_data_from_json()
    print(data.head())
    print(data.tail())
