
import utility.nltk_downloader as nltk_downloader

import nltk
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from bs4 import BeautifulSoup


def download_resource() -> None:

    """
    Downloads the required NLTK components.
    """

    nltk_downloader.fetch('punkt')
    nltk_downloader.fetch('stopwords')
    nltk_downloader.fetch('gutenberg')
    nltk_downloader.fetch('shakespeare')


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

    return word_tokenize(str(text).lower())


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


def clean_tokens(text: str) -> list[str]:

    """
    Get a cleaned (without HTML tags) list of tokens without stopwords.

    :param text: Raw text to clean.
    :return: Return a list of clean words.
    """

    return remove_stopwords(tokenize_text(remove_html_tags(text)), is_stem=False)


def clean_stemmed_tokens(text: str) -> list[str]:

    """
    Get a cleaned (without HTML tags) list of stemmed tokens without stopwords.

    :param text: Raw text to clean.
    :return: Return a list of clean stemmed words.
    """

    return remove_stopwords(tokenize_text(remove_html_tags(text)), is_stem=True)


def clean_stemmed_text(text: str) -> str:

    """
    Get a cleaned (without HTML tags) list of stemmed tokens without stopwords.

    :param text: Raw text to clean.
    :return: Return a string of clean stemmed words.
    """

    return ' '.join(clean_stemmed_tokens(text=text))


'''

def clean_message(message, stemmer=PorterStemmer(),
                  stop_words=set(stopwords.words('english'))):
    # Converts to Lower Case and splits up the words
    words = word_tokenize(message.lower())

    filtered_words = []

    for word in words:
        # Removes the stop words and punctuation
        if word not in stop_words and word.isalpha():
            filtered_words.append(stemmer.stem(word))

    return filtered_words


# Challenge: Modify function to remove HTML tags. Then test on Email with DOC_ID 2.
def clean_msg_no_html(message, stemmer=PorterStemmer(),
                      stop_words=set(stopwords.words('english'))):
    # Remove HTML tags
    soup = BeautifulSoup(message, 'html.parser')
    cleaned_text = soup.get_text()

    # Converts to Lower Case and splits up the words
    words = word_tokenize(cleaned_text.lower())

    filtered_words = []

    for word in words:
        # Removes the stop words and punctuation
        if word not in stop_words and word.isalpha():
            filtered_words.append(stemmer.stem(word))
    #             filtered_words.append(word)

    return filtered_words
'''
