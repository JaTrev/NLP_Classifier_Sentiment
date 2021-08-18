import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # explicitly require this experimental feature
from sklearn.model_selection import HalvingRandomSearchCV
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# for text pre-processing
import re
import string
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

    return data


@st.cache(allow_output_mutation=True)
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


def simple_preprocess(text):
    # convert to lowercase, strip and remove punctuations
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

    return TweetTokenizer().tokenize(text)


def remove_stopwords(tokens: list, dynamic_flag: bool = False, counter: Counter = None, max_value: int = -1):

    if dynamic_flag:

        assert isinstance(counter, Counter)
        assert max_value >= 0

        tokens = [i for i in tokens if counter[i] <= max_value]

    processed_tokens = [i for i in tokens if i not in stopwords.words('english')]

    return processed_tokens


def do_lemmatization(tokens: list):

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
        counter = None

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


def do_tf_idf_vectorization(data: list, test_data: list = None):
    assert isinstance(data[0], str)

    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    data_tfidf = tfidf_vectorizer.fit_transform(data)

    if test_data:
        test_data_tfidf = tfidf_vectorizer.transform(test_data)
    else:
        test_data_tfidf = None

    return data_tfidf.toarray().tolist(), tfidf_vectorizer.get_feature_names(), test_data_tfidf.toarray().tolist()


def do_tf_vectorization(data: list, test_data: list = None):
    assert isinstance(data[0], str)

    tf_vectorizer = CountVectorizer()
    data_tf = tf_vectorizer.fit_transform(data)

    if test_data:
        test_data_tf = tf_vectorizer.transform(test_data)
    else:
        test_data_tf = None

    return data_tf.toarray().tolist(), tf_vectorizer.get_feature_names(), test_data_tf.toarray().tolist()


def train_w2v_model(tokenized_data: list):
    return Word2Vec(tokenized_data, min_count=1)


def do_w2v_vectorization(training_tweets: pd.DataFrame, testing_tweets: pd.DataFrame, testing_labels: pd.DataFrame):

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


@st.cache
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


def do_sentiment_analysis(df: pd.DataFrame, sentiment_type: str = "Vader"):

    if sentiment_type == "Vader":
        analyzer = SentimentIntensityAnalyzer()

    return df.processed_text.apply(lambda x: analyzer.polarity_scores(x)["compound"]).array.reshape(-1, 1)


def do_vectorization(training_df: pd.DataFrame, y_training: pd.DataFrame, testing_df: pd.DataFrame,
                     y_testing: pd.DataFrame, feature: str):
    assert feature in ['Bag of Words', 'TF-IDF', 'Word Embeddings', 'Sentiment Analysis']

    if feature == "Bag of Words":
        bow_model = train_bow_model(training_df)

        training_vectors = get_bow_vectors(bow_model, tokens=training_df.processed_text)
        testing_vectors = get_bow_vectors(bow_model, tokens=testing_df.processed_text)

    elif feature == "TF-IDF":
        tf_idf_model = train_bow_model(training_df, use_tf_idf=True)

        training_vectors = get_bow_vectors(tf_idf_model, tokens=training_df.processed_text)
        testing_vectors = get_bow_vectors(tf_idf_model, tokens=testing_df.processed_text)

    elif feature == "Word Embeddings":
        training_temp_df, testing_temp_df = do_w2v_vectorization(training_df, testing_df, y_testing)

        training_vectors = training_temp_df.text_vector.values.tolist()
        testing_vectors = testing_temp_df.text_vector.values.tolist()

        y_testing = testing_temp_df.target.values.tolist()

    else:
        # feature == "Sentiment Analysis"

        training_vectors = do_sentiment_analysis(training_df)
        testing_vectors = do_sentiment_analysis(testing_df)

    return training_vectors, y_training, testing_vectors, y_testing


def get_model(model_name: str, x: list, y: list):
    assert model_name == "rf"
    assert len(x) == len(y)

    parms = {'n_estimators': 30, 'min_samples_split': 8, 'min_samples_leaf': 5, 'max_features': 50, 'max_depth': 100,
             'bootstrap': True, 'class_weight': "balanced"}
    rf = RandomForestClassifier(**parms)  # LogisticRegression()
    rf.fit(x, y)
    return rf


def do_classification(x_train: list, y_train: list, x_test: list, y_test: list, model_name: str = "rf"):

    assert len(x_train) == len(y_train), f"mismatch in training data: " \
                                                  f"{len(x_train)} vs. {len(y_train)}"
    assert len(x_test) == len(y_test), f"mismatch in testing data: {len(x_test)} vs. {len(y_test)}"

    model = get_model(model_name=model_name, x=x_train, y=y_train)

    y_train_predict = model.predict(x_train)
    y_test_predict = model.predict(x_test)

    train_score = accuracy_score(y_true=y_train, y_pred=y_train_predict)
    test_score = accuracy_score(y_true=y_test, y_pred=y_test_predict)

    return train_score, test_score


def perform_grid_search(classifier_name: str, x_train: list, y_train: list, x_test: list, y_test: list):
    assert classifier_name == "rf", f"Error: have not selected a valid classification model: {classifier_name}"

    model = RandomForestClassifier()
    param_grid = {
        'bootstrap': [True],
        'max_depth': [3, 5, 10, 30, 40, 50, 60, 80, 90, 100],
        'min_samples_leaf': [3, 4, 5, 10, 20, 40],
        'min_samples_split': [8, 10, 12, 20, 40],
        'n_estimators': [30, 50, 100, 200, 300, 1000, 1500],
        'max_features': ['auto', 'log2', 50, 100]
    }

    grid_search = HalvingRandomSearchCV(estimator=model, param_distributions=param_grid, cv=5, n_jobs=-1)

    # perform grid search
    grid_search.fit(x_train, y_train)

    # predict training and test
    y_predict_training = grid_search.best_estimator_.predict(x_train)
    y_predict_test = grid_search.best_estimator_.predict(x_test)

    # calculate train and accuracy scores
    train_score = accuracy_score(y_true=y_train, y_pred=y_predict_training)
    test_score = accuracy_score(y_true=y_test, y_pred=y_predict_test)


    return grid_search.best_estimator_, grid_search.best_params_, train_score, test_score
