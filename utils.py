import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# for text pre-processing
import re, string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer, word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from gensim.models import Word2Vec

from collections import Counter


# convert to lowercase, strip and remove punctuations
def simple_preprocess(text):
    text = text.lower()
    text = text.strip()
    text = re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]' ,' ' ,text)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    tknzr = TweetTokenizer()
    return tknzr.tokenize(text)


# stopword removal
def remove_stopwords(tokens, dynamic_flag=False, counter:Counter=None, max_value=-1):

    if dynamic_flag:

        assert isinstance(counter, Counter)
        assert max_value >= 0
        processed_tokens = [i for i in tokens if counter[i] <= max_value]

    processed_tokens = [i for i in tokens if i not in stopwords.words('english')]

    return processed_tokens


# lemmatization
def do_lemmatization(tokens):
    wl = WordNetLemmatizer()
    processed_tokens = [wl.lemmatize(w) for w in tokens]
    return processed_tokens


@st.cache
def do_preprocessing(list_of_tweets: list, stop_word_flag=True,
                     stop_word_dynamic_flag=False, lemmatization_flag=True, max_value=0):

    list_of_tweets = [simple_preprocess(tweet) for tweet in list_of_tweets]

    processed_tweets = []
    if stop_word_dynamic_flag:

        counter = Counter([w for tweet in list_of_tweets for w in tweet])

    else:
        counter =None

    for tweet in list_of_tweets:

        tweet_tokens = tweet

        if stop_word_flag or stop_word_dynamic_flag:
            tweet_tokens = remove_stopwords(tweet_tokens, dynamic_flag=stop_word_dynamic_flag,
                                            counter=counter, max_value=max_value)

        if lemmatization_flag:
            tweet_tokens = do_lemmatization(tweet_tokens)

        processed_tweets.append(" ".join(tweet_tokens))

    return processed_tweets


def do_TF_IDF_vectorization(data: list, test_data:list = None):
    assert isinstance(data[0], str)

    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    data_tfidf = tfidf_vectorizer.fit_transform(data)

    if test_data:
        test_data_tfidf = tfidf_vectorizer.transform(test_data)
    else:
        test_data_tfidf = None

    return data_tfidf.toarray().tolist(), tfidf_vectorizer.get_feature_names(), test_data_tfidf.toarray().tolist()


def do_TF_vectorization(data: list, test_data:list = None):
    assert isinstance(data[0], str)

    tf_vectorizer = CountVectorizer()
    data_tf = tf_vectorizer.fit_transform(data)

    if test_data:
        test_data_tf = tf_vectorizer.transform(test_data)
        # test_data_tf = test_data_tf.todense()
    else:
        test_data_tf = None

    print("testing vectors")
    training_vectors = data_tf.toarray().tolist()
    print(training_vectors[:2])

    return data_tf.toarray().tolist(), tf_vectorizer.get_feature_names(), test_data_tf.toarray().tolist()


@st.cache
def get_w2v_model(tokenized_data: list):

    model = Word2Vec(tokenized_data, min_count=1)

    return model


# @st.cache
def do_W2V_vectorization(training_df: pd.DataFrame, testing_df: pd.DataFrame):

    tokenized_data = [word_tokenize(sent) for sent in training_df["processed text"]]
    model = get_w2v_model(tokenized_data)

    training_sentence_w2v = [np.mean([model.wv.get_vector(w) for w in words], axis=0) for words in tokenized_data]

    testing_df = testing_df.reset_index()
    tokenized_test_data = [word_tokenize(sent) for sent in testing_df["processed text"]]

    rows = []
    for i_d, tokenized_d in enumerate(tokenized_test_data):

        word_vectors = [model.wv.get_vector(w) for w in tokenized_d if w in model.wv.index2word]

        d_w2v = []
        if len(word_vectors) > 1:
            d_w2v = np.mean(word_vectors, axis=0)

        elif len(word_vectors) == 1:
            d_w2v = word_vectors[0]

        if len(d_w2v):
            row = testing_df.iloc[i_d].values.tolist()
            row.append(d_w2v)
            rows.append(row)

    new_columns = testing_df.columns.tolist()
    new_columns.append('text_vector')

    new_testing_df = pd.DataFrame(rows)
    new_testing_df.columns = new_columns
    new_testing_df.set_index("id", inplace=True)

    new_training_df = training_df
    new_training_df["text_vector"] = training_sentence_w2v

    return new_training_df, new_testing_df
