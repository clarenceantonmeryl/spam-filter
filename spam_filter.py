import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import recall_score, precision_score, f1_score

import data_pre_processor

DATA_JSON_FILE = 'data/source/data-stemmed.json'


def predict():
    data = pd.read_json(DATA_JSON_FILE)
    data.sort_index(inplace=True)
    # print(data)
    vectorizer = CountVectorizer(stop_words='english')

    all_features = vectorizer.fit_transform(data.MESSAGE)
    # print(all_features)
    # print(all_features.shape)

    # Vocabulary
    vocabulary = vectorizer.vocabulary_
    # print(vocabulary)

    x_train, x_test, y_train, y_test = train_test_split(all_features, data.CATEGORY, test_size=0.3, random_state=88)

    # print(x_train.shape)
    # print(x_test.shape)

    classifier = MultinomialNB()
    classifier.fit(x_train, y_train)

    correct_classification = (y_test == classifier.predict(x_test)).sum()
    print(f'{correct_classification} documents classified correctly')

    incorrect_classification = y_test.size - correct_classification
    print(f'Number of documents incorrectly classified is {incorrect_classification}')

    fraction_wrong = incorrect_classification / (correct_classification + incorrect_classification)
    print(f'The (testing) accuracy of the model is {1 - fraction_wrong:.2%}')

    classifier_accuracy = classifier.score(x_test, y_test)
    print("Classifier Score: ", classifier_accuracy)

    recall_ratio = recall_score(y_test, classifier.predict(x_test))
    print("Recall Score: ", recall_ratio)

    precision_ratio = precision_score(y_test, classifier.predict(x_test))
    print("Precision Score: ", precision_ratio)

    f1_harmonic_mean = f1_score(y_test, classifier.predict(x_test))
    print("F1 Score: ", f1_harmonic_mean)

    input_document = [
        'get drugs for free now!',
        'need a mortgage? Reply to arrange a call with a specialist and get a quote',
        'hi anton how about we meet in the morning and drive my car for to the service station',
        'Hello Jonathan, how about a game of golf tomorrow?',
        'Ski jumping is a winter sport in which competitors aim to achieve the longest jump after descending from a '
        'specially designed ramp on their skis. Along with jump length, competitor\'s style and other factors affect '
        'the final score. Ski jumping was first contested in Norway in the late 19th century, and later spread '
        'through Europe and North America in the early 20th century. Along with cross-country skiing, it constitutes '
        'the traditional group of Nordic skiing disciplines. '
    ]

    stemmed_document = [data_pre_processor.clean_stemmed_text(doc) for doc in input_document]
    print(stemmed_document)

    input_term_matrix = vectorizer.transform(stemmed_document)
    # print(input_term_matrix)

    prediction = classifier.predict(input_term_matrix)

    print("PREDICTION", prediction)
