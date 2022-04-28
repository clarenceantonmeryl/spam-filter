import numpy as np
import pandas as pd
from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt
import nltk

import data_processor

WHALE_FILE = 'resource/wordcloud/whale-icon.png'
SKULL_FILE = 'resource/wordcloud/skull-icon.png'
THUMBS_UP_FILE = 'resource/wordcloud/thumbs-up.png'
THUMBS_DOWN_FILE = 'resource/wordcloud/thumbs-down.png'

FONT_FILE = 'resource/wordcloud/OpenSansCondensed-Bold.ttf'


def get_mask_file(name='WHALE'):
    match name:
        case 'WHALE':
            return WHALE_FILE
        case 'SKULL':
            return SKULL_FILE
        case 'THUMBS_UP':
            return THUMBS_UP_FILE
        case 'THUMBS_DOWN':
            return THUMBS_DOWN_FILE
        case default:
            return None


def generate_wordcloud(text: str, mask='WHALE', colormap='ocean'):

    icon = Image.open(get_mask_file(name=mask))
    image_mask = Image.new(mode='RGB', size=icon.size, color=(255, 255, 255))
    image_mask.paste(icon, box=icon)
    rgb_array = np.array(image_mask)

    # https://matplotlib.org/stable/tutorials/colors/colormaps.html

    word_cloud = WordCloud(
        mask=rgb_array,
        background_color='white',
        colormap=colormap,
        max_words=600,
        font_path=FONT_FILE,
        max_font_size=300,
    ).generate(text.upper())

    plt.figure(figsize=[16, 8])
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


def generate_email_wordcloud(data: pd.DataFrame):

    """
    Generate word cloud for spam and ham messages.

    :param data: A DataFrame of messages.
    """

    doc_ids_spam = data[data.CATEGORY == 1].index
    doc_ids_ham = data[data.CATEGORY == 0].index

    nested_list_all = data.MESSAGE.apply(data_processor.clean_tokens)  # Make sure it is not stemmed.
    nested_list_spam = nested_list_all.loc[doc_ids_spam]
    nested_list_ham = nested_list_all.loc[doc_ids_ham]

    flat_list_spam = [word for sub_list in nested_list_spam for word in sub_list]
    flat_list_ham = [word for sub_list in nested_list_ham for word in sub_list]

    ham_text = ' '.join(flat_list_ham)
    spam_text = ' '.join(flat_list_spam)

    generate_wordcloud(text=ham_text, mask='THUMBS_UP', colormap='winter')
    generate_wordcloud(text=spam_text, mask='THUMBS_DOWN', colormap='gist_heat')

    # generate_wordcloud(text=ham_text, mask='THUMBS_UP', colormap='Greens')
    # generate_wordcloud(text=spam_text, mask='THUMBS_DOWN', colormap='Reds')


def get_gutenberg_novel(name='melville-moby_dick'):
    story = nltk.corpus.gutenberg.words(f'{name}.txt')
    word_list = [''.join(word) for word in story]
    novel_text = ' '.join(word_list)
    return novel_text


def sample_machine_learning():
    generate_wordcloud(text="Machine learning (ML) is the study of computer algorithms that can improve automatically "
                            "through experience and by the use of data.[1] It is seen as a part of artificial "
                            "intelligence. Machine learning algorithms build a model based on sample data, "
                            "known as training data, in order to make predictions or decisions without being "
                            "explicitly programmed to do so.[2] Machine learning algorithms are used in a wide "
                            "variety of applications, such as in medicine, email filtering, speech recognition, "
                            "and computer vision, where it is difficult or unfeasible to develop conventional "
                            "algorithms to perform the needed tasks. A subset of machine learning is closely related "
                            "to computational statistics, which focuses on making predictions using computers; but "
                            "not all machine learning is statistical learning. The study of mathematical optimization "
                            "delivers methods, theory and application domains to the field of machine learning. Data "
                            "mining is a related field of study, focusing on exploratory data analysis through "
                            "unsupervised learning.[5][6] Some implementations of machine learning use data and "
                            "neural networks in a way that mimics the working of a biological brain.[7][8] In its "
                            "application across business problems, machine learning is also referred to as predictive "
                            "analytics.")


def sample_melville():
    novel_text = get_gutenberg_novel(name='melville-moby_dick')
    generate_wordcloud(text=novel_text, mask='WHALE', colormap='ocean')


def sample_hamlet():
    novel_text = get_gutenberg_novel(name='shakespeare-hamlet')
    generate_wordcloud(text=novel_text, mask='SKULL', colormap='bone')


# sample_melville()
# sample_hamlet()
