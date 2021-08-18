from utils import *
import streamlit as st
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff


def set_up():
    st.set_page_config("NLP Classifier by Jason Thies", layout="centered", initial_sidebar_state="expanded")

    text_chapter = """
                    <style>.chapter {
                    font-size:40px ;font-family: 'Cooper Black'; color: grey;} 
                    </style>
                    """
    st.markdown(text_chapter, unsafe_allow_html=True)

    text_subchapter = """
                        <style>.subchapter {
                        font-size:30px ;font-family: 'Cooper Black'; color: grey;} 
                        </style>
                        """
    st.markdown(text_subchapter, unsafe_allow_html=True)

    title_font = """
               <style>.title {
               font-size:50px ; font-family: 'Cooper Black'; color: #FF9633;} 
               </style>
               """
    st.markdown(title_font, unsafe_allow_html=True)

    text_font = """
                <style>.text {
                 font-family: 'Cooper Black'; color: black;} 
                </style>
                """

    error_font = """
                    <style>.warning {
                     font-family: 'Cooper Black'; color: red;} 
                    </style>
                    """
    st.markdown(error_font, unsafe_allow_html=True)
    st.markdown(text_font, unsafe_allow_html=True)

    page_style = """
        <style>
        /* This is to hide hamburger menu completely */
        #MainMenu {visibility: hidden;}

        /* This is to hide Streamlit footer */
        footer {visibility: hidden;}
        """
    st.markdown(page_style, unsafe_allow_html=True)


def markdown_text(text: str, text_class: str = "text"):

    st.markdown(f'<p class="{text_class}"> {text} </p>', unsafe_allow_html=True)


def introduction():

    st.markdown('<p class="title">NLP Classifier</p>', unsafe_allow_html=True)

    st.markdown('<p class="text">'
                'In this project, we train a natural language processing (NLP) classifier_name. '
                'The project describes the standard pipeline for creating a traditional classifier_name, '
                'from preprocessing the textual data to deploy a trained model.'
                '</p>', unsafe_allow_html=True)


def data_description(dataset_name: str, dataset: pd.DataFrame):

    subset_1 = dataset[dataset.target == 1]

    markdown_text("0. Dataset", text_class="chapter")

    markdown_text("In the left sidebar, you can choose the dataset that we will work with. "
                  f'Currently, we are working with the "{dataset_name}". '
                  f"This dataset includes Tweets about real-world disasters and Tweets that do not discuss disasters. "
                  f"Each tweet has a unique identifier and was hand-labeled to one of the two classes. "
                  f"The dataset includes {len(dataset)} tweets, "
                  f"{len(subset_1)} ({round(100*(len(subset_1)/len(dataset)), 2)}%) "
                  f"of these tweets are labeled as disaster tweets. All other tweets are random, non-disaster tweets. "
                  f"The following table shows a snippet of the dataset, including tweets and their labels "
                  f"(1:= disaster tweet).")


def get_database():
    dataset_container = st.sidebar.beta_expander("Dataset", True)

    with dataset_container:
        dataset = st.selectbox("Choose a dataset", ("Select a dataset.", "Twitter Disaster Corpus"))

        if dataset == "Twitter Disaster Corpus":
            st.markdown("The 'Twitter Disaster Corpus' dataset includes tweets, "
                        "each tweet was hand-labeled and is classified as a disaster tweet or as a non-disaster tweet.")

    if dataset == "Select a dataset.":
        return None
    else:
        return dataset


def preprocessing_selector():
    container = st.sidebar.beta_expander("Preprocessing")

    preprocessing_options = container.multiselect('Select method(s)',
                                    ['Static Stop Word Removal', 'Lemmatization', 'Dynamic Stop Word Removal'],
                                    help="Select the preprocessing method(s) to be used on the original text.")

    if 'Dynamic Stop Word Removal' in preprocessing_options:
        max_freq = container.slider(
            "Maximum word count",
            min_value=1,
            max_value=20,
            value=10,
            step=4,
        )

    else:
        max_freq = None

    markdown_text("2. Preprocessing Tweets", text_class="chapter")

    markdown_text("After splitting the dataset, the training and validation set are preprocessed individually. "
                  "This prevents data leakage, which occurs when information that would not be available at "
                  "prediction time is used during training. "
                  "Data leakage causes the model to perform well on the validation set but not during deployment.")

    markdown_text('In the sidebar, please select a preprocessing technique to be applied to the tweets. '
                  'Note, before applying any of these techniques, tokenization is applied to the tweets. '
                  'Tokenization is a common preprocessing technique that splits text (a large string) '
                  'into a list of smaller strings. '
                  'Paragraphs can be tokenized into sentences and sentences can be tokenized into words. '
                  'Furthermore, all hyperlinks are removed and tweet is lower-cased.')

    if len(preprocessing_options):
        show_processed_data = True
        preprocessing_techniques = ", ".join(preprocessing_options)

        markdown_text(f'The following preprocessing techniques are applied to the tweets: {preprocessing_techniques}.')

    else:
        show_processed_data = False
        markdown_text("Please select a preprocessing technique in the sidebar.", text_class="warning")

    if "Static Stop Word Removal" in preprocessing_options or "Dynamic Stop Word Removal" in preprocessing_options:
        text = 'Stop word removal is the process of removing all stop words from the text. '\
               'Stop words refer to the most common words in a language, e.g. "the", "is", and "on" are stop words. '\
               'In addition, these words often do not hold semantic value or indicate the sentiment of a sentence, ' \
               'hence we can remove these words before training our model to save computing time. '

        if "Static Stop Word Removal" in preprocessing_options:
            text += "Static stop word removal is a sentiment_type of stop word removal that uses a " \
                    "predefined stop word list, all words from this list are removed from our tweets."

        if "Dynamic Stop Word Removal" in preprocessing_options:
            text += "Dynamic stop word removal is a sentiment_type of stop word removal that is based on " \
                    "term frequency and does not use a predefined stop word list."

        markdown_text("Stop Word Removal:", text_class="subchapter")
        markdown_text(text)

    if "Lemmatization" in preprocessing_options:
        markdown_text("Lemmatization:", text_class="subchapter")
        markdown_text('Lemmatization is a normalization approach that transforms all words into their respective lemma.'
                      ' Lemma is the canonical form of a set of words, '
                      'e.g.: "broken", and "breaking" share the same lemma (break). '
                      'Note we do not include stemming as a normalization approach in this project. '
                      'Although it is a common approach, '
                      'stemming can produce non-meaningful words that do not exist in the dictionary, '
                      'e.g.: "studies" is transformed to "studi".')

    if show_processed_data:
        markdown_text('In the following data snippet, '
                      'we can see the difference between preprocessed tweets and the original tweets.')

    return preprocessing_options, max_freq, show_processed_data


def show_data(df: pd.DataFrame, column_keys: list, column_names: list, column_width: list, first_n_rows: int = 5,
              table_placeholder=None):

    assert all([c in df.columns for c in column_keys])
    assert len(column_names) == len(column_keys)
    assert len(column_width) == len(column_names)

    df_show_table = df[column_keys][:first_n_rows]
    df_show_table.columns = column_names

    if table_placeholder is not None:
        table_placeholder = table_placeholder.table(df_show_table)
    else:
        table_placeholder = st.table(df_show_table)

    return table_placeholder


def visualize_feature(x: list, y: list, feature_name: str):

    y_values = set(y)
    x_per_y_value = [[x[i] for i, y_value in enumerate(y_values) if y == y_value] for y in [0, 1]]
    fig = ff.create_distplot(x_per_y_value, ["No Disaster", "Disaster"], show_curve=True)

    fig.update(layout_title_text=f'Density of {feature_name}')
    st.plotly_chart(fig)


def visualize_score(train_score: list, test_score: list):
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"sentiment_type": "indicator"}] * 2],
    )

    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=train_score,
        title={"text": f"Training Accuracy"},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={"axis": {"range": [0, 1]}},
        delta={"reference": train_score}),
        row=1, col=1
    )

    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=test_score,
        title={"text": f"Testing Accuracy"},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={"axis": {"range": [0, 1]}},
        delta={"reference": test_score}),
        row=1, col=2
    )
    st.plotly_chart(fig)


def do_dataset_split(df: pd.DataFrame, target_column: str = "target"):

    markdown_text("1. Splitting the labeled data", text_class="chapter")

    markdown_text("The first step is to split the labeled data into a training set and a validation set (80:20).  "
                  "The classifier_name trains on the training set,"
                  " automatically updating its parameters to improve do_classification of the samples in this set. "
                  "The validation set is used to evaluate the model's performance.")

    st.image("pics/training_validate.png",
             caption="Figure 2: Splitting the labeled data into training and validation",
             width=500)

    X = df.drop(target_column, axis=1)
    y = df.target

    # split data into training and test
    return train_test_split(X, y, test_size=.3, random_state=42, stratify=y)


def do_feature_engineering():

    markdown_text("3. Feature Engineering", text_class="chapter")

    markdown_text("After preprocessing the tweets we can now perform feature engineering. "
                  "This step is one of the most important steps in machine learning. "
                  "In NLP, the text-based models need to have numerical features. "
                  "These features can be simple, such as expressing the number of words in a sentence, "
                  "but we can also use more complex feature extraction techniques like Bag of Words (BOW) or "
                  "word embeddings. "
                  "Extracting the sentiment of a sentence can also be an additional feature that helps our "
                  "classifier improve its performance. ")

    container = st.sidebar.beta_expander("Feature Engineering")

    feature = container.selectbox('Select a feature to calculate',
                                  ['No feature has been selected.', 'Bag of Words', 'TF-IDF', 'Word Embeddings',
                                   'Sentiment Analysis'],
                                  help="Select a feature on which the classifier_name will predict class labels.")

    if feature == "No feature has been selected.":
        markdown_text("Select a feature used to train the classifier_name.", text_class="warning")

    else:
        markdown_text(f"Currently we are using {feature} to represent the text in numbers.")

        if feature == "Bag of Words":
            # bow
            markdown_text("Bag of Words(BoW)", text_class="subchapter")
            markdown_text("The Bag of Words (BoW) model is the simplest form of text representation in numbers. "
                          "The technique works by first creating a vocabulary of all unique words from the dataset. "
                          "Then each word occurrence is marked in each data sample, creating vectors with 0s and 1s. ")

        elif feature == "TF-IDF":
            # tf-idf
            markdown_text("TF-IDF", text_class="TF-IDF")
            markdown_text('TF-IDF creates a BoW model but calculates the term frequency-inverse document frequency '
                          '(IF-IDF) value instead of the absolute frequency. '
                          'This statistic reflects how important a word is to a tweet in the dataset. '
                          'TF-IDF counts the absolute frequency of a word in the tweet and '
                          'divides it by the number of tweets it occurs in. ')

        elif feature == "Word Embeddings":
            # word embedding
            markdown_text("Word Embeddings", text_class="subchapter")
            markdown_text('A word embedding is a representation of a word where words that have similar meanings '
                          'have similar presentations. This approach follows the distributional hypothesis, '
                          'words that have a similar context will have similar meanings. '
                          'Unlike BoW, word embeddings are dense word representations learned based on context.  '
                          'Many different word embedding techniques exist. This project uses Word2Vec embeddings.')

        else:
            # feature == "Sentiment Analysis"
            markdown_text("Sentiment Analysis", text_class="subchapter")
            markdown_text("Sentiment Analysis feature has not yet been implemented", text_class="warning")

    if feature in ["No feature has been selected.", "Sentiment Analysis"] :
        return None
    else:
        return feature


def get_classifier(feature_name: str):
    model_dic = {"Random Forest": "rf"}
    grid_search_flag = False

    markdown_text("4. Classification", text_class="chapter")
    markdown_text(f"With the new feature ({feature_name}), we can now perform train a classification. "
                  f"The classifier is trained using the training data and assessed using the validation dataset. ")

    with st.sidebar.beta_expander("Classification"):
        model = st.selectbox('Select a classification model',
                             ['No model has been selected.', 'Random Forest'],
                             help="Select classification to be used with the feature calculated.")

    if model == 'No model has been selected.':
        return None, grid_search_flag
    else:
        markdown_text(f"Currently the a {model} classifier is being used.")
        return model_dic[model], grid_search_flag


if __name__ == "__main__":
    set_up()
    introduction()
    dataset = get_database()

    if dataset is None:
        markdown_text("Select a dataset in the left sidebar to get started.")

    else:

        all_data = get_data(dataset)
        data_description(dataset, all_data)
        table_placeholder = show_data(all_data, column_keys=["text", "target"], column_names=["Tweet", "Class"],
                                      column_width=[500, 200])

        X_train, X_valid, y_train, y_valid = do_dataset_split(all_data)

        options, max_freq, do_processing = preprocessing_selector()

        if do_processing:

            # split preprocessing to prevent data leakage
            X_train = preprocess_data(X_train, options, max_freq)
            X_valid = preprocess_data(X_valid, options, max_freq)

            column_keys = ["processed_text", "text"]
            column_names = ["Processed Text", "Text"]
            column_width = [1000, 1000]

            table_placeholder = show_data(X_train, column_keys=column_keys, column_names=column_names,
                                          column_width=column_width)

        feature = do_feature_engineering()

        if feature is not None:

            if do_processing:
                X_train, y_train, X_test, y_test = do_vectorization(training_df=X_train, y_training=y_train,
                                                                    testing_df=X_valid, y_testing=y_valid,
                                                                    feature=feature)

                # do_classification
                model, do_grid_search = get_classifier(feature_name=feature)
                if model is not None:

                    if do_grid_search:
                        _, _, train_score, test_score = perform_grid_search(model, x_train=X_train,
                                                                            y_train=y_train, x_test=X_test,
                                                                            y_test=y_test)
                    else:
                        train_score, test_score = do_classification(x_train=X_train, y_train=y_train,
                                                                    x_test=X_test, y_test=y_test,
                                                                    model_name=model)

                    visualize_score(train_score=train_score, test_score=test_score)
                else:
                    st.markdown('<p class="warning">'
                                'Select a model to perform classification.'
                                '</p>', unsafe_allow_html=True)

            else:
                st.markdown('<p class="warning">'
                            'Need to preprocess the data before the classifier_name can be trained.'
                            '</p>', unsafe_allow_html=True)
