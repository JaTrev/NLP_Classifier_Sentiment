import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# for text pre-processing
import re, string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer, word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from collections import Counter
nltk.download('wordnet')


@st.cache(allow_output_mutation=True)
def get_data(data: str):

    if data == "Twitter Disaster Corpus":
        data = pd.read_csv("train.csv", index_col="id")[["text", "target"]]

    print("len of data")
    print(len(data))
    print(data.describe())
    print(data.head(-1))

    return data


@st.cache
def preprocess_data(df: pd.DataFrame, options, max_freq):
    new_df = df.copy(deep=True)

    stop_word_flag, stop_word_dynamic_flag, lemmatization_flag = get_flags(options)

    tweets = new_df["text"].values.tolist()

    new_df["processed_text"] = do_preprocessing(tweets, stop_word_flag, stop_word_dynamic_flag,
                                                lemmatization_flag, max_value=max_freq)

    return new_df


@st.cache
def get_flags(options: list):
    # Stop Word Removal', 'Lemmatization', 'Dynamic Stop Word Removal

    stop_word_flag, stop_word_dynamic_flag, lemmatization_flag = False, False, False
    if "Stop Word Removal" in options:
        stop_word_flag = True

    if "Dynamic Stop Word Removal" in options:
        stop_word_dynamic_flag = True

    if "Lemmatization" in options:
        lemmatization_flag = True

    return stop_word_flag, stop_word_dynamic_flag, lemmatization_flag


# convert to lowercase, strip and remove punctuations
def simple_preprocess(text):
    text = text.lower()
    text = text.strip()
    text = re.sub(r'https?:\/\/\S*', '', text, flags=re.MULTILINE)
    text = re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', ' ', text)
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


@st.cache
def basic_features(tweets_list: list, feature: str):

    assert feature in ['word count', 'char count', 'VADER_sentiment', 'text_vector']

    tweets = pd.Series(tweets_list)

    if feature == "word count":
        return tweets.apply(lambda x: len(str(x).split(" ")))

    elif feature == "char count":
        return tweets.apply(lambda x: sum(len(word) for word in str(x).split(" ")))

    elif feature == "VADER_sentiment":
        analyzer = SentimentIntensityAnalyzer()
        return tweets.apply(lambda x: analyzer.polarity_scores(x)["compound"])
    else:
        print("error")


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
def train_w2v_model(tokenized_data: list):

    model = Word2Vec(tokenized_data, min_count=1)

    return model


# @st.cache
def do_W2V_vectorization(training_tweets: pd.DataFrame, testing_tweets: pd.DataFrame, testing_labels: pd.DataFrame):

    testing_labels_list = testing_labels.values.tolist()
    tokenized_data = [word_tokenize(sent) for sent in training_tweets.processed_text]
    model = train_w2v_model(tokenized_data)

    training_sentence_w2v = [np.mean([model.wv.get_vector(w) for w in words], axis=0) for words in tokenized_data]

    testing_tweets = testing_tweets.reset_index()
    tokenized_test_data = [word_tokenize(sent) for sent in testing_tweets.processed_text]

    rows = []
    for i_d, tokenized_d in enumerate(tokenized_test_data):

        word_vectors = [model.wv.get_vector(w) for w in tokenized_d if w in model.wv.index2word]

        d_w2v = []
        if len(word_vectors) > 1:
            d_w2v = np.mean(word_vectors, axis=0)

        elif len(word_vectors) == 1:
            d_w2v = word_vectors[0]

        if len(d_w2v):
            row = testing_tweets.iloc[i_d].values.tolist()
            row.append(d_w2v)
            row.append(testing_labels_list[i_d])
            rows.append(row)

    new_columns = testing_tweets.columns.tolist()
    new_columns.extend(['text_vector', 'target'])

    new_testing_df = pd.DataFrame(rows)
    new_testing_df.columns = new_columns
    new_testing_df.set_index("id", inplace=True)

    new_training_df = training_tweets
    new_training_df["text_vector"] = training_sentence_w2v

    assert len(training_tweets) == len(new_training_df)
    return new_training_df, new_testing_df


def train_bow_model(tokens: pd.DataFrame, use_tf_idf: bool = False):

    def dummy(temp_tokens):
        return temp_tokens

    if use_tf_idf:
        model = TfidfVectorizer(
            tokenizer=dummy,
            preprocessor=None,
            lowercase=False
        )

    else:
        model = CountVectorizer(
            tokenizer=dummy,
            preprocessor=None,
            lowercase=False,
        )

    return model.fit(tokens)


def get_bow_vectors(bow_model: CountVectorizer, tokens: pd.DataFrame):
    return bow_model.transform(tokens).toarray()


def do_vectorization(training_df: pd.DataFrame, y_training: pd.DataFrame, testing_df: pd.DataFrame,
                     y_testing: pd.DataFrame, feature: str):
    assert feature in ['Bag of Words', 'TF-IDF', 'Word Embeddings']

    if feature == "Bag of Words":
        bow_model = train_bow_model(training_df)

        training_vectors = get_bow_vectors(bow_model, tokens=training_df.processed_text)
        testing_vectors = get_bow_vectors(bow_model, tokens=testing_df.processed_text)

    elif feature == "TF-IDF":
        tf_idf_model = train_bow_model(training_df, use_tf_idf=True)

        training_vectors = get_bow_vectors(tf_idf_model, tokens=training_df.processed_text)
        testing_vectors = get_bow_vectors(tf_idf_model, tokens=testing_df.processed_text)

    elif feature == "Word Embeddings":
        training_temp_df, testing_temp_df = do_W2V_vectorization(training_df, testing_df, y_testing)

        training_vectors = training_temp_df.text_vector.values.tolist()
        testing_vectors = testing_temp_df.text_vector.values.tolist()

        y_testing = testing_temp_df.target.values.tolist()
    else:
        assert feature == "sentiment"
        print("TODO")


    return training_vectors, y_training, testing_vectors, y_testing

