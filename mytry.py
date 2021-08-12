import pandas as pd
from utils import do_preprocessing, do_TF_vectorization, do_TF_IDF_vectorization, do_W2V_vectorization
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler

import plotly.graph_objs as go


def introduction():
    st.title("**Welcome My Playground**")

    st.subheader("This is a place where you can get familiar with machine learning models directly from your browser")

    st.text(""
            "1. Choose a dataset\n"
    )


def dataset_selector():
    dataset_container = st.sidebar.beta_expander("Dataset", True)

    with dataset_container:
        dataset = st.selectbox("Choose a dataset", ("Twitter Disaster Corpus", None))

        n_samples = st.number_input(
            "Number of samples",
            min_value=50,
            max_value=1000,
            step=10,
            value=300,
        )

    return dataset, n_samples


def preprocessing_selector():

    options = st.multiselect('Preprocessing options',
                             ['Stop Word Removal', 'Lemmatization', 'Dynamic Stop Word Removal'],
                             help="Select the processing method(s) to be used.")

    if 'Dynamic Stop Word Removal' in options:
        max_freq = st.slider(
                "Maximum word count",
                min_value=1,
                max_value=20,
                value=10,
                step=4,
        )

    else:
        max_freq = None

    if len(options):
        show_processed_data = True

    else:
        show_processed_data = False

    return options, max_freq, show_processed_data


def show_preprocessing(all_data: pd.DataFrame, title="Raw Data", first_n_rows: int = 20):

    title_placeholder = st.text('Loading data...')

    options, max_freq, do_processing = preprocessing_selector()

    show_columns = ["text", "target"]
    df_show_table = all_data[show_columns][:first_n_rows]

    # st.dataframe(all_data.reset_index()[show_columns][:first_n_rows])

    if not do_processing:
        title_placeholder.title(title)

        column_keys = ["text", "target"]
        column_names = ["Text", "Target"]
        column_width = [1400, 150]

        # plotly_table = go.Table(
        #    columnorder=[1, 2],
        #    columnwidth=[1400, 150],
        #    header=dict(values=["Text", "Target"], fill_color='paleturquoise', align='left'),
        #    cells=dict(values=[df_show_table.text, df_show_table.target], fill_color='lavender', align='left')
        # )

    else:
        stop_word_flag, stop_word_dynamic_flag, lemmatization_flag = get_flags(options)

        tweets = all_data["text"].values.tolist()
        all_data["processed_text"] = do_preprocessing(tweets, stop_word_flag, stop_word_dynamic_flag,
                                                      lemmatization_flag, max_value=max_freq)

        title_placeholder.title("Processed Data")

        column_keys = ["processed_text", "text"]
        column_names = ["Processed Text", "Text"]
        column_width = [1000, 1000]

        #df_show_table = all_data[show_columns][:first_n_rows]
        #plotly_table = go.Table(
        #    columnorder=[1, 2],
        #    columnwidth=[1000, 1000],
        #    header=dict(values=["Processed Text", "Original Text"], fill_color='paleturquoise', align='left'),
        #    cells=dict(values=[df_show_table.processed_text, df_show_table.text], fill_color='lavender', align='left')
        #)

    #fig = go.Figure(data=[plotly_table])
    #st.plotly_chart(fig, use_container_width=True)

    table_placeholder = show_data(all_data, column_keys=column_keys, column_names=column_names,
                                  column_width=column_width)
    return all_data, table_placeholder


def show_data(df: pd.DataFrame, column_keys: list, column_names: list, column_width: list, first_n_rows: int = 5,
              table_placeholder=None):

    assert all([c in df.columns for c in column_keys])
    assert len(column_names) == len(column_keys)
    assert len(column_width) == len(column_names)

    # show_columns = ["processed_text", "text"]
    df_show_table = df[column_keys][:first_n_rows]
    plotly_table = go.Table(
        columnorder=[1, 2],
        columnwidth=column_width,
        header=dict(values=column_names, fill_color='paleturquoise', align='left'),
        cells=dict(values=[df_show_table[key] for key in column_keys], fill_color='lavender', align='left')
    )

    fig = go.Figure(data=[plotly_table])

    if table_placeholder is not None:
        table_placeholder = table_placeholder.plotly_chart(fig, use_container_width=True)

    else:
        table_placeholder = st.plotly_chart(fig, use_container_width=True)

    return table_placeholder




@st.cache
def get_data(is_training_data: bool = True):

    if is_training_data:
        data = pd.read_csv("train.csv", index_col="id")

    else:
        assert is_training_data is False
        data = pd.read_csv("train.csv", index_col="id")

    return data


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


def feature_engineering(tweets: pd.DataFrame):

    word_count = tweets.apply(lambda x: len(str(x).split(" ")))
    char_count = tweets.apply(lambda x: sum(len(word) for word in str(x).split(" ")))

    avg_word_length = char_count / word_count

    analyzer = SentimentIntensityAnalyzer()
    sentiment = tweets.apply(lambda x: analyzer.polarity_scores(x)["compound"])

    return word_count, char_count, avg_word_length, sentiment


def feature_vis(dataframe:pd.DataFrame):
    # Create distplot with custom bin_size

    if st.checkbox('Show bivariate distribution'):

        feature_name = st.text_input('Feature Name', 'VADER_sentiment', max_chars=15)

        if feature_name in dataframe.columns:
            x, y = feature_name, "target"
            fig, ax = plt.subplots(nrows=1, ncols=2)
            fig.suptitle(x, fontsize=12)
            for i in [0, 1]:
                sns.distplot(dataframe[dataframe[y] == i][x], hist=True, kde=False,
                             bins=10, hist_kws={"alpha": 0.8},
                             axlabel="histogram", ax=ax[0])

                sns.distplot(dataframe[dataframe[y] == i][x], hist=False, kde=True,
                             kde_kws={"shade": True}, axlabel="density",
                             ax=ax[1])
            ax[0].grid(True)
            ax[0].legend(["Not Disaster", "Disaster"])
            ax[1].grid(True)
            fig.set_size_inches(10, 2)

            st.pyplot(fig)
            # plt.show()
        else:
            st.subheader("Feature was not found.\n"
                         "Make sure the feature is a column name in the data set.")


def do_vectorization(training_df: pd.DataFrame, testing_df: pd.DataFrame):

    if True:

        classify_by_x = st.radio('Classify by:', ['word count', 'char count', 'avg word count',
                                                           'VADER_sentiment', 'text_vector'])

        if classify_by_x in set(training_df.columns):

            training_vectors = training_df[[classify_by_x]].values.tolist()
            testing_vectors = testing_df[[classify_by_x]].values.tolist()

            y_train = training_df["target"]
            y_test = testing_df["target"]

        else:
            assert classify_by_x == "text_vector"
            training_temp_df, testing_temp_df = do_W2V_vectorization(training_df, testing_df)

            training_vectors = training_temp_df[classify_by_x].values.tolist()
            testing_vectors = testing_temp_df[classify_by_x].values.tolist()

            y_train = training_temp_df["target"]
            y_test = testing_temp_df["target"]

        return training_vectors, y_train, testing_vectors, y_test


def classification(training_vecs, training_y, testing_vecs, testing_y):
    assert len(training_vecs) == len(training_y), f"mismatch in training data: " \
                                                  f"{len(training_vecs)} vs. {len(training_y)}"
    assert len(testing_vecs) == len(testing_y), f"mismatch in testing data: {len(testing_vecs)} vs. {len(testing_y)}"

    lr_w2v = LogisticRegression()
    lr_w2v.fit(training_vecs, training_y)

    # Predict y value for test dataset
    y_predict = lr_w2v.predict(testing_vecs)

    test_acc = accuracy_score(y_true=testing_y, y_pred=y_predict)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=test_acc,
        title={"text": f"Accuracy (test)"},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={"axis": {"range": [0, 1]}},
        delta={"reference": test_acc},
    ))

    st.plotly_chart(fig)


if __name__ == "__main__":
    st.set_page_config("Page Title", layout="wide")

    dataset, n_samples = dataset_selector()

    all_data = get_data(dataset)[:n_samples]

    all_data, table_placeholder = show_preprocessing(all_data)

    training_data = all_data.sample(frac=0.8, random_state=42)
    test_data = all_data.drop(training_data.index)

    if st.checkbox('Apply Feature Engineering') and "processed_text" in training_data.columns:
        word_count, char_count, avg_word_length, sentiment = feature_engineering(training_data["processed_text"])
        training_data["word count"] = word_count
        training_data["char count"] = char_count
        training_data["avg word count"] = avg_word_length
        training_data["VADER_sentiment"] = sentiment

        word_count, char_count, avg_word_length, sentiment = feature_engineering(test_data["processed_text"])
        test_data["word count"] = word_count
        test_data["char count"] = char_count
        test_data["avg word count"] = avg_word_length
        test_data["VADER_sentiment"] = sentiment

        show_data(training_data,
                  column_names=["Processed Text", "Text", "Word Count", "Char Count", "Avg Word Count", "Sentiment"],
                  column_keys=["processed_text", "text", "word count", "char count", "avg word count",
                               "VADER_sentiment"],
                  column_width=[1000, 500, 200, 200, 200, 200],
                  table_placeholder=table_placeholder)

        # show_preprocessing(training_data[["processed text", "text", "word count", "char count",
        # "avg word count", "VADER_sentiment"]], title="Feature Engineering")


        feature_vis(training_data)

        X_train, y_train, X_test, y_test = do_vectorization(training_data, test_data)
        # classification
        classification(training_vecs=X_train, training_y=y_train, testing_vecs=X_test, testing_y=y_test)












