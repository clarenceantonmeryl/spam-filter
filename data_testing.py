import numpy as np

import data_loader

PROB_SPAM = 0.311


def testing_model():
    print("TESTING BEGINS")
    prob_tokens_spam, prob_tokens_ham, prob_tokens_all, x_test, y_test = data_loader.load_test_models()

    print(prob_tokens_spam.shape, prob_tokens_ham.shape, prob_tokens_all.shape, x_test.shape, y_test.shape)

    # print(x_test.shape)
    # print(prob_tokens_spam.shape)
    # dot_product = x_test.dot(prob_tokens_spam)
    # print(dot_product)
    # print(dot_product.shape)

    '''
    # Find and replace any probability of value 0
    small_probability = 1
    large_probability = 0

    index = 0;
    for probability in prob_tokens_all:
        if probability < small_probability:
            print(index, probability)
            small_probability = probability
        if probability > large_probability:
            large_probability = probability
        index += 1

    print(small_probability, large_probability)
    '''

    joint_log_spam = x_test.dot(np.log(prob_tokens_spam) - np.log(prob_tokens_all)) + np.log(PROB_SPAM)
    # joint_log_spam = x_test.dot(np.log(prob_tokens_spam)) + np.log(PROB_SPAM)
    print(joint_log_spam)

    joint_log_ham = x_test.dot(np.log(prob_tokens_ham) - np.log(prob_tokens_all)) + np.log(1 - PROB_SPAM)
    # joint_log_ham = x_test.dot(np.log(prob_tokens_ham)) + np.log(1 - PROB_SPAM)
    print(joint_log_ham)

    prediction = (joint_log_spam > joint_log_ham) * 1

    count = 0
    for index in range(len(prediction)):
        if prediction[index] != y_test[index]:
            print(index, prediction[index], y_test[index])
            count += 1
    print("Total false predictions:", count, "Total", len(prediction))

