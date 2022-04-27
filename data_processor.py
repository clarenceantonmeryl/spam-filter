
import utility.nltk_downloader as nltk_downloader

import nltk
from nltk.stem import PorterStemmer
# from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from bs4 import BeautifulSoup


def download_resource():
    nltk_downloader.fetch('punkt')
    nltk_downloader.fetch('stopwords')


# Converting to lower case
# Tokenizing
# Removing stop words
# Word stemming
# Removing HTML tags


stemmer = PorterStemmer()
# stemmer = SnowballStemmer('english')


def stem_word(word):
    return stemmer.stem(word)


def tokenize_message(message):
    return word_tokenize(message.lower())


def remove_stopwords(words_list):
    stopwords_set = set(stopwords.words('english'))
    return [stem_word(word) for word in words_list if word.isalpha() and word not in stopwords_set]


def remove_html_tags(message):
    soup = BeautifulSoup(message, 'html.parser')
    # return soup.prettify()
    return soup.get_text()

