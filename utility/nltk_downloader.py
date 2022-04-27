import nltk
import ssl


def fetch(resource):
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        print("AttributeError")
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download(resource)
    # nltk.download('punkt')
    # nltk.download('stopwords')


# fetch('punkt')
# fetch('stopwords')