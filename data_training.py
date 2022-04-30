import pandas as pd
import numpy as np
import utility.constant as constant


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


def generate_trained_models(sparse_train_data, sparse_test_data):

    full_train_data = make_full_matrix(sparse_train_data)

    print(full_train_data)

    # Training the Naive Bayes Model

    # Calculating the Probability of Spam

    print("A", full_train_data.shape)
    print("B", full_train_data.CATEGORY.size)
    print("C", full_train_data[full_train_data.CATEGORY == 1].shape)
    print("D", full_train_data[full_train_data.CATEGORY == 0].shape)

    total_emails_in_training_data = full_train_data.shape[0]
    spam_emails_in_training_data = full_train_data[full_train_data.CATEGORY == 1].shape[0]
    probability_spam = spam_emails_in_training_data / total_emails_in_training_data
    print("Total emails", total_emails_in_training_data)
    print("Spam emails", spam_emails_in_training_data)
    print("Probability of Spam", probability_spam)

    # Total number of words

    full_train_features = full_train_data.loc[:, full_train_data.columns != 'CATEGORY']
    print("Features\n", full_train_features)
    email_lengths = full_train_features.sum(axis=1)
    print("email_length\n", email_lengths)
    print("email_length.shape", email_lengths.shape)
    total_word_count = email_lengths.sum()
    print("Total Words", total_word_count)

    # Number of words in spam and ham emails.

    spam_lengths = email_lengths[full_train_data.CATEGORY == 1]
    print("spam_lengths\n", spam_lengths)
    print("spam_lengths.shape", spam_lengths.shape)
    spam_word_count = spam_lengths.sum()
    print("spam_word_count", spam_word_count)

    ham_lengths = email_lengths[full_train_data.CATEGORY == 0]
    print("ham_lengths\n", ham_lengths)
    print("ham_lengths.shape", ham_lengths.shape)
    ham_word_count = ham_lengths.sum()
    print("ham_word_count", ham_word_count)

    print("Check", total_word_count - spam_word_count - ham_word_count)  # = 0

    print("Average number of words in spam emails = ", spam_word_count / spam_lengths.shape[0])
    print("Average number of words in ham emails = ", ham_word_count / ham_lengths.shape[0])

    # Summing the words / tokens occuring in spam

    print(full_train_features.shape)

    # train_spam_tokens
    spam_train_features = full_train_features.loc[full_train_data.CATEGORY == 1]
    print("train_spam_tokens\n", spam_train_features)

    # summed_spam_tokens
    summed_spam_features = spam_train_features.sum(axis=0) + 1  # Laplace Smoothing : to avoid division by 0
    print("summed_spam_features\n", summed_spam_features)

    # train_spam_tokens
    ham_train_features = full_train_features.loc[full_train_data.CATEGORY == 0]
    print("train_spam_tokens\n", ham_train_features)

    # summed_spam_tokens
    summed_ham_features = ham_train_features.sum(axis=0) + 1  # Laplace Smoothing : to avoid division by 0
    print("summed_spam_features\n", summed_ham_features)
    print("Check", ham_train_features[2499].sum() + 1)

    # P(Token | Spam) - Probability that a Token Occurs given the Email is spam

    prob_tokens_spam = summed_spam_features / (spam_word_count + constant.VOCABULARY_SIZE)
    print("prob_tokens_spam\n", prob_tokens_spam)
    print("prob_tokens_spam sum = 1\n", prob_tokens_spam.sum())

    # P(Token | Ham) - Probability that a Token Occurs given the Email is non-spam

    prob_tokens_ham = summed_ham_features / (ham_word_count + constant.VOCABULARY_SIZE)
    print("prob_tokens_ham\n", prob_tokens_ham)
    print("prob_tokens_ham sum = 1\n", prob_tokens_ham.sum())

    # P(Token) - Probability that token occurs

    prob_tokens_all = full_train_features.sum(axis=0) / total_word_count
    print("prob_tokens_all\n", prob_tokens_all)
    print("prob_tokens_all sum = 1\n", prob_tokens_all.sum())

    print(sparse_test_data.shape)

    full_test_data = make_full_matrix(sparse_test_data)
    # full_test_features
    x_test = full_test_data.loc[:, full_test_data.columns != 'CATEGORY']
    y_test = full_test_data.CATEGORY

    print(x_test)
    print(y_test)

    return prob_tokens_spam, prob_tokens_ham, prob_tokens_all, x_test, y_test
