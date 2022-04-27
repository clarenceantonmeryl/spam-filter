
import utility.nltk_downloader as nltk_downloader

import nltk
from nltk.stem import PorterStemmer
# from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from bs4 import BeautifulSoup


def download_resource() -> None:

    """
    Downloads the required NLTK components.
    """

    nltk_downloader.fetch('punkt')
    nltk_downloader.fetch('stopwords')


# Converting to lower case
# Tokenizing
# Removing stop words
# Word stemming
# Removing HTML tags


stemmer = PorterStemmer()
# stemmer = SnowballStemmer('english')


def stem_word(word: str) -> str:

    """
    Stem the word.

    :param word: The word to be stemmed.
    :return: Stemmed word.
    """
    return stemmer.stem(word)


def tokenize_text(text) -> list[str]:

    """
    Tokenize the given text.

    :param text: Text to be tokenized.
    :return: List of tokenized words.
    """

    return word_tokenize(text.lower())


def get_stopwords() -> set[str]:

    """
    Get the set of stopwords.

    :return: A set of stopwords.
    """

    return set(stopwords.words('english'))


def remove_stopwords(words_list: list[str], is_stem=True) -> list:

    """
    Remove stopwords from the list of words.

    :param is_stem: Flag to stem the words
    :param words_list: List of words to filter the stopwords.
    :return: A filtered list of words with the stopwords.
    """

    stopwords_set = get_stopwords()
    if is_stem:
        return [stem_word(word) for word in words_list if word.isalpha() and word not in stopwords_set]
    else:
        return [word for word in words_list if word.isalpha() and word not in stopwords_set]


def remove_html_tags(html_text: str) -> str:

    """
    Remove the HTML tags from the text.

    :param html_text: Text with HTML tags.
    :return: Text without HTML tags.
    """

    soup = BeautifulSoup(html_text, 'html.parser')
    # return soup.prettify()
    return soup.get_text()


def clean_stemmed_tokens(text: str, is_stem=True) -> list[str]:

    """
    Get a cleaned (without HTML tags) list of stemmed tokens without stopwords.

    :param is_stem: Flag to apply stemming.
    :param text: Raw text to clean.
    :return: Return a list of clean stemmed words.
    """

    return remove_stopwords(tokenize_text(remove_html_tags(text)), is_stem)




