import logging
import pickle
from pprint import pprint
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict, Counter
from os import listdir
from os.path import splitext
import time
import re
import warnings

from enum import Enum

from gensim.models import Word2Vec
from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore

from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords

from wordcloud import WordCloud
from bs4 import BeautifulSoup

from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score

def setup_logging():
    """
    Setup logging to file and to output.
    Clears up previous active logging.
    """
    # Clear potential previous mess
    logging.shutdown()

    # Configure logging to log to file and to output.
    logging.basicConfig(
        format='%(asctime)s:%(name)s:%(levelname)s:	 %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG,
        handlers=[logging.FileHandler("main.log"), logging.StreamHandler()])

    # Logging all set up and ready to be used for this run.
    logging.info('--------Starting a fresh run-----------')

def get_timestamp():
    t = time.localtime(time.time())
    return "%02d%02d%02d%02d%02d" % (t.tm_mon,
                                     t.tm_mday,
                                     t.tm_hour,
                                     t.tm_min,
                                     t.tm_sec)


def save_model(model, model_name):
    logging.info("Saving model: %s", model_name)
    file_name = model_name + ".pkl"
    with open(file_name, 'wb') as model_pkl:
        pickle.dump(model, model_pkl)

def load_model(model_name):
    logging.info("Loading model: %s", model_name)
    file_name = model_name + ".pkl"
    with open(file_name, 'rb') as model_pkl:
        model = pickle.load(model_pkl)
    return model


def save_results(result, file_name="result"):
    """
    Requires result to be a pandas Data Frame type.
    Saves it to a .csv file.
    """
    relevant_results = result[[
        'mean_test_acc',
        'mean_test_f1_micro',
        'mean_test_neg_log_loss',
        'mean_test_prec_micro',
        'mean_test_rec_micro',
        'std_test_acc',
        'std_test_f1_micro',
        'std_test_neg_log_loss',
        'std_test_prec_micro',
        'std_test_rec_micro',
        'params'
        ]]
    full_name = file_name + get_timestamp() + ".csv"
    relevant_results.to_csv(full_name)
    logging.info("%s saved as %s", file_name, full_name)

def load_csv(file_name):
    """
    Loads a .csv file and returns the content as a list.
    Each element in the list contains one row from the .csv file.
    """
    logging.info('Loading csv file: %s', file_name)
    data_frame = pd.read_csv(file_name, dtype={'type': np.dtype(str), 'posts': np.dtype(str)})

    logging.info('Loading complete')
    return data_frame


def handle_delimiter(text):
    """
    Function that handles the delimiter of posts (|||).
    It replaces the delimiter with a space.
    """
    # TODO: Maybe we should actually split here, in the case of sentiment analysis of sentences?
    text = re.sub(r'\|\|\|', " ", str(text)) # Replace ||| with spaces
    return str(text)

def to_lower(text):
    """
    Function that turns text to lower-case.
    """
    return str(text).lower()


def clean_posts(text):
    """
    Function that runs Beautiful Soup to remove keywords not specific for texts.
    Also removes any URL found and replaces it with <URL> tag.
    """
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'http\S+', r'<URL>', str(text)) # Replace URLs with URL tag.
    # Optional: Remove @ and replace with some tag?
    # Optional: Instead of replacing URL with URL tag, do URL analysis and turn it into keywords?
    return str(text)

def preprocess_posts(posts):
    """
    Function that takes a collection of strings and
    returns a collection containing lists with processed posts.

    Input:
        "posts" is a collection of strings.
        Each string contain 50 posts, separated by "|||".
        The string does not always contain "'" as a first letter, but usually it does.
    Output:
        Collection where the elements are lists of processed posts.
    Processing:
        The string containing the posts has the prefix and suffix "'" removed,
        and is split on "|||".
    """
    logging.info("Preprocessing data...")
    processed_posts = []
    for string in posts:
        start = 1 if string[0] == "'" else 0
        end = -1 if string[-1] == "'" else None
        trimmed = str(string[start:end])
        cleaned = str(clean_posts(trimmed))
        processed_posts.append(cleaned)
    logging.info("Preprocessing data completed.")
    return processed_posts


def plot_types(types):
    """ Plot barplot of all MBTI types in the data set. """
    count_types = types.value_counts()

    plt.figure(figsize=(12, 4))
    sns.set_color_codes("pastel")
    sns.barplot(count_types.index, count_types.values, alpha=0.8)
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Types', fontsize=12)
    plt.savefig("types_barplot.pdf", bbox_inches='tight')


def plot_subtypes(types):
    """ Plot barplot split on subtypes (I/E, N/S, T/F, J/P). """
    subtypes = ["I", "E", "N", "S", "T", "F", "J", "P"]
    subtype_counts = defaultdict(int)
    for mbti_type, count in types.value_counts().iteritems():
        for subtype in subtypes:
            if subtype in mbti_type:
                subtype_counts[subtype] += count
    data_frame = pd.DataFrame.from_dict(subtype_counts, "index")
    print(data_frame)
    return # TODO: Do nice plotting somehow


def process_media_links(posts):
    """
    A lot of posts contain URLs, which are meaningless if not processed.
    My idea is that:
        youtube URLs can be processed into a few keywords describing the clip.
        Images can be reverse searched and keywords perhaps extracted.
    """
    pass


def plot_media_link_types(posts_collection):
    """
    Helper function that plots the type of media links in the training data set.
    Types could, for example, be:
        youtube
        vimeo
        tumblr
    """
    websites = ["youtube", "youtu.be", "vimeo", "tumblr", "spotify", "jpg", "png", "gif", "jpeg", "images", "personalitycafe"]
    counts = defaultdict(int)
    pattern = re.compile(r"http([^\s]+)")
    for posts in posts_collection:
        for post in posts:
            #print(post)
            result = pattern.search(post)
            if result:
                url = result.group()
                for website in websites:
                    if website in url:
                        counts[website] += 1
                        break
                else:
                    print(url)
    print(counts)

def plot_learning_curve(X, y, train_sizes, train_scores, test_scores, title='', ylim=None, figsize=(14,8)):
    plt.figure(figsize=figsize)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def create_wordclouds(train):
    """
    Takes a Pandas DataFrame containing posts for a MBTI type.
    Creates a global wordcloud and one wordcloud for each MBTI type present in the data set.
    """
    logging.info("Processing data for wordclouds...")
    words_for_type = defaultdict(str)
    all_words = ""
    for mbti_type, posts in train.values:
        all_words += str(posts) + " "
        words_for_type[mbti_type] += str(posts) + " "

    generate_wordcloud(all_words, "wordclouds/wordcloud_all")

    for mbti_type, words in words_for_type.items():
        generate_wordcloud(words, "wordclouds/wordcloud_{}".format(mbti_type))
    logging.info("All wordclouds generated!")


def generate_wordcloud(text, wordcloud_name="wordcloud"):
    """
    Helper function that takes a text and creates a wordcloud for that text.
    The wordcloud is save with the supplied name.
    """
    logging.info("Generating wordcloud: %s...", wordcloud_name)
    text = handle_delimiter(text)
    # Generate a word cloud image
    wordcloud = WordCloud().generate(text)

    # lower max_font_size
    #wordcloud = WordCloud(max_font_size=40).generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("{}.pdf".format(wordcloud_name), bbox_inches='tight')
    logging.info("Wordcloud generated!")


def get_data(argv):
    """
    Function that loads the raw MBTI data set and preprocesses it.
    Returns the processed data set.
    """
    logging.info("Loading raw data.")
    train = load_csv(DATA_FILE)

    logging.info("Using loaded, preprocessed data.")
    train["posts"] = preprocess_posts(train["posts"])

    return train


def grid_search(model, parameters, scoring, refit="f1_micro"):
    kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    logging.info("Using kfolds: %s", kfolds)
    cv_clf = GridSearchCV(model, parameters, scoring=scoring, refit=refit, cv=kfolds,
                          n_jobs=-1, verbose=10, return_train_score=True)
    return cv_clf

def model_helper(model_name, X_train, y_train, steps, parameters, scoring):
    """
    Helper function that creates the pipeline and performs grid search on it.
    Logs and saves results, returns the best model.
    """
    pipeline = Pipeline(steps)
    logging.info("Pipeline set up: %s.", pipeline.named_steps)

    clf = grid_search(pipeline, parameters, scoring)

    clf.fit(X_train, y_train)
    logging.info("Best index: %s, with score: %s", clf.best_index_, clf.best_score_)

    results = pd.DataFrame.from_dict(clf.cv_results_)
    logging.info(results)
    save_results(results, model_name + "_result")

    save_model(clf, model_name)
    return clf

def get_model(model_name, argv, X_train, y_train, scoring, prob_scoring):
    if "--load-model" in argv:
        return load_model(model_name)

    if model_name == "lr":
        return train_logistic_regression_model(X_train, y_train, {**scoring, **prob_scoring})

    if model_name == "etc":
        return train_extra_trees_model(X_train, y_train, {**scoring, **prob_scoring})

    if model_name == "sgd":
        return train_linear_sgd_model(X_train, y_train, scoring)

    raise Exception("Model name {} is unknown!".format(model_name))

def train_extra_trees_model(X_train, y_train, scoring):
    """
    Function that creates a pipeline with:
        1. TFIDF vectorization.
        2. Truncated SVD.
        3. Extra Trees Classifier.

    It performs GridSearchCV on the pipeline, 
    choosing the model with the best (cross-validated) f1_micro score.

    The CV scores are saved to a separate .csv file (timestamped) and
    the best model is returned.
    """
    steps = [
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('tsvd', TruncatedSVD()),
        ('extra_trees', ExtraTreesClassifier(n_jobs=-1))]

    parameters = {
        'tfidf__max_df': [0.577],
        'tsvd__n_components': [25],
        'extra_trees__n_estimators': [200],
        'extra_trees__class_weight': ["balanced"]
    }

    return model_helper("extra_trees", X_train, y_train, steps, parameters, scoring)


def train_logistic_regression_model(X_train, y_train, scoring):
    """
    Function that setup a Logistic Regression model.
    It performs GridSearchCV on the pipeline,
    choosing the model with the best (cross-validated) f1_micro score.

    The CV scores are saved to a separate .csv file (timestamped) and
    the best model is returned.
    """
    #steps = [
    #    ('tfidf', CountVectorizer(stop_words='english')),
    #    ('lr', LogisticRegression(n_jobs=1))]

    #parameters = {
    #    'tfidf__max_features': [None],
    #    'tfidf__max_df': [0.577],
    #    'tfidf__ngram_range': [(1, 1)],
    #    
    #}
    steps = [
        ('tfidf', TfidfVectorizer(max_features=40000, sublinear_tf=True)),
        ('lr', LogisticRegression(n_jobs=1))
    ]

    parameters = {
        'lr__solver': ["lbfgs"],
        'lr__multi_class': ["multinomial"],
        'lr__class_weight': ["balanced"],
        'lr__C': [0.005]
    }
    return model_helper("lr", X_train, y_train, steps, parameters, scoring)


def train_linear_sgd_model(X_train, y_train, scoring):
    """
    Function that creates a linear model trained with SGD.
    It performs GridSearchCV on the pipeline,
    choosing the model with the best (cross-validated) f1_micro score.

    The CV scores are saved to a separate .csv file (timestamped) and
    the best model is returned.
    """
    steps = [
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('svm', SGDClassifier())]

    parameters = {
        'tfidf__max_df': [0.577],
        'tfidf__ngram_range': [(1, 1)]
    }
    return model_helper("sgd", X_train, y_train, steps, parameters, scoring)


def tokenize_and_stem(data, types, create_corpus=False, filter_level=None):
    logging.info("Tokenizing data...")
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize_sents(data)
    logging.info("Tokenizing done.")

    logging.info("Removing stop words...")
    stop_words = stopwords.words('english')

    if filter_level == "types" or filter_level == "extremes":
        stop_words.extend([t.lower() for t in types])
    print(stop_words)

    relevant_tokens = [[word for word in token if word not in stop_words]
                       for token in tokens]
    logging.info("Stop words removed.")

    logging.info("Stemming data...")
    stemmer = EnglishStemmer()
    stemmed_tokens = [[stemmer.stem(token) for token in tokens]
                      for tokens in relevant_tokens]
    logging.info("Data stemmed.")

    if create_corpus:
        dictionary = corpora.Dictionary()
        dictionary.add_documents(stemmed_tokens)
        if filter_level == "extremes":
            dictionary.filter_extremes()

        dictionary.save(DICTIONARY_FILE)  # store the dictionary, for future reference

        # Convert to bag-of-words representation
        corpus = [dictionary.doc2bow(tokens) for tokens in stemmed_tokens]
        corpora.MmCorpus.serialize(CORPUS_FILE, corpus)  # store to disk, for later use
    texts = [" ".join(token) for token in stemmed_tokens]
    return texts

def get_topic_features(X_set, lda, dictionary):
    docs_topics = [lda.get_document_topics(dictionary.doc2bow(doc.split())) for doc in X_set]

    def doc_to_features(doc):
        features = np.zeros(NUM_TOPICS)
        for index, prob in doc:
            features[index] = prob
        return features
        
    return [doc_to_features(doc) for doc in docs_topics]

def get_char_features(X_set):
    chars = list("abcdefghijklmnopqrstuvwxyz.,!?:;)( ")
    def doc_to_featues(doc):
        return Counter(doc)
    char_counts = [doc_to_featues(doc) for doc in X_set]

    def char_count_to_feature(char_count):
        feature_value = np.zeros(len(chars))
        for char, count in char_count.items():
            if char in chars:
                feature_value[chars.index(char)] = count
        return feature_value

    return [char_count_to_feature(char_count) for char_count in char_counts]


def get_term_features(X_set, lda, dictionary):
    docs_topics = [lda.get_document_topics(dictionary.doc2bow(doc.split())) for doc in X_set]

    def doc_to_features(doc, doc_topics):
        indices = dictionary.doc2idx(doc.split())
        word_counts = Counter(indices)
        features = np.zeros(NUM_TOPICS * 10, dtype=int)
        for index, prob in doc_topics:
            n_words = int(round(prob * 10))
            if n_words > 0:
                word_ids_probs = lda.get_topic_terms(index, n_words)
                for i, (word_id, word_prob) in enumerate(word_ids_probs):
                    features[(index * 10) + i] = 1 + word_counts[word_id]
        return features

    return [doc_to_features(doc, doc_topics) for doc, doc_topics in zip(X_set, docs_topics)]


def get_features(X_set, feature_type, lda, dictionary):
    if feature_type == "chars":
        return get_char_features(X_set), "{}".format(feature_type)
    if feature_type == "terms":
        return get_term_features(X_set, lda, dictionary), "{}_{}".format(feature_type, NUM_TOPICS)
    if feature_type == "topics":
        return get_topic_features(X_set, lda, dictionary), "{}_{}".format(feature_type, NUM_TOPICS)
    if feature_type == "raw":
        return X_set, feature_type
    raise Exception("Feature type {} is unknown!".format(feature_type))

def get_terms_classifier(classifier, scoring, prob_scoring):
    lin_params = {
        "base": {
            "penalty": ['l1'],
            "loss": ["squared_hinge"],
            "max_iter": [500]
        },
        100: {"C": [0.3116]},
        75: {"C": [0.1775]},
        50: {"C": [0.214]},
        25: {"C": [0.5335]},
        16: {"C": [0.3866]},
        10: {"C": [0.0736]}
    }

    logit_params = {
        "base": {
            'solver': ["lbfgs"],
            'multi_class': ["multinomial"],
            'class_weight': ["balanced"],
        },
        100:{'C': [0.005]},
        75:{'C': [1.2]},
        50:{'C': [0.05]},
        25:{'C': [0.15]},
        16:{'C': [0.005]},
        10:{'C': [0.05]}
    }

    etc_params = {
        'max_features': [10],
        'criterion': ["gini"],
        'class_weight': ["balanced"],
        'n_estimators': [200]
    }

    adaboost_params = {
        100: {'n_estimators': [7]},
        75: {'n_estimators': [7]},
        50: {'n_estimators': [5]},
        25: {'n_estimators': [6]},
        16: {'n_estimators': [10]},
        10: {'n_estimators': [7]}
    }

    gradboost_params = {
        "base": {
            "learning_rate": [0.1],
            "n_estimators": [200],
            "max_depth": [1]
        }
    }
    
    if classifier == "linear":
        params = {**lin_params["base"], **lin_params[NUM_TOPICS]}
        model = LinearSVC(dual=False, random_state=1907)
        scorings = scoring
    elif classifier == "logit":
        params = {**logit_params["base"], **logit_params[NUM_TOPICS]}
        model = LogisticRegression(random_state=1907)
        scorings = {**scoring, **prob_scoring}
    elif classifier == "etc":
        params = etc_params
        model = ExtraTreesClassifier(n_jobs=-1, random_state=1907)
        scorings = {**scoring, **prob_scoring}
    elif classifier == "adaboost":
        params = adaboost_params[NUM_TOPICS]
        model = AdaBoostClassifier(random_state=1994)
        scorings = {**scoring, **prob_scoring}
    elif classifier == "gradboost":
        params = gradboost_params["base"]
        model = GradientBoostingClassifier(random_state=1994)
        scorings = {**scoring, **prob_scoring}
    else:
        raise Exception("Classifier {} is unknown!".format(classifier))
    return grid_search(model, params, scorings)

def get_topics_classifier(classifier, scoring, prob_scoring):
    lin_params = {
        "base": {
            "loss": ["squared_hinge"],
            "penalty": ['l1'],
            "max_iter": [100]
        },
        100: {"C": [0.355]},
        75: {"C": [0.8]},
        50: {"C": [0.5]},
        25: {"C": [0.5]},
        16: {"C": [0.2]},
        10: {"C": [0.2]}
    }

    logit_params = {
        "base": {
            'solver': ["lbfgs"],
            'multi_class': ["multinomial"],
            'class_weight': ["balanced"]
        },
        100:{'C': [0.125]},
        75:{'C': [2]},
        50:{'C': [2]},
        25:{'C': [2]},
        16:{'C': [0.005]},
        10:{'C': [1]},
    }

    etc_params = {
        "base": {
            'max_features': [10],
            'criterion': ["gini"],
            'class_weight': ["balanced"]
        },
        100:{'n_estimators': [700]},
        75:{'n_estimators': [700]},
        50:{'n_estimators': [700]},
        25:{'n_estimators': [700]},
        16:{'n_estimators': [700]},
        10:{'n_estimators': [700]}
    }

    adaboost_params = {
        100: {'n_estimators': [7]},
        75: {'n_estimators': [10]},
        50: {'n_estimators': [9]},
        25: {'n_estimators': [10]},
        16: {'n_estimators': [5]},
        10: {'n_estimators': [9]}
    }

    gradboost_params = {
        "base": {
            "learning_rate": [0.1],
            "n_estimators": [200],
            "max_depth": [1]
        }
    }
    
    if classifier == "linear":
        params = {**lin_params["base"], **lin_params[NUM_TOPICS]}
        model = LinearSVC(dual=False, random_state=1907)
        scorings = scoring
    elif classifier == "logit":
        params = {**logit_params["base"], **logit_params[NUM_TOPICS]}
        model = LogisticRegression(random_state=1907)
        scorings = {**scoring, **prob_scoring}
    elif classifier == "etc":
        params = {**etc_params["base"], **etc_params[NUM_TOPICS]}
        model = ExtraTreesClassifier(n_jobs=-1, random_state=1907)
        scorings = {**scoring, **prob_scoring}
    elif classifier == "adaboost":
        params = adaboost_params[NUM_TOPICS]
        model = AdaBoostClassifier(random_state=1994)
        scorings = {**scoring, **prob_scoring}
    elif classifier == "gradboost":
        params = gradboost_params["base"]
        model = GradientBoostingClassifier(random_state=1994)
        scorings = {**scoring, **prob_scoring}
    else:
        raise Exception("Classifier {} is unknown!".format(classifier))
    return grid_search(model, params, scorings)

def get_chars_classifier(classifier, scoring, prob_scoring):
    # Do not normalize.
    lin_params = {
        "penalty": ['l1'],
        "loss": ["squared_hinge"],
        "max_iter": [500],
        "C": [0.1103]
    }

    # Do not normalize
    logit_params = {
        'solver': ["lbfgs"],
        'multi_class': ["multinomial"],
        'class_weight': ["balanced"],
        'C': [0.784]
    }

    # Normalize!
    etc_params = {
        'n_estimators': [700],
        'max_features': [10],
        'criterion': ["gini"],
        'class_weight': ["balanced"]
    }

    # Untrained
    adaboost_params = {
        'n_estimators': [13]
    }

    gradboost_params = {
        "learning_rate": [0.3],
        "n_estimators": [100],
        "max_depth": [1]
    }
    
    if classifier == "linear":
        params = lin_params
        model = LinearSVC(dual=False, random_state=1907)
        scorings = scoring
    elif classifier == "logit":
        params = logit_params
        model = LogisticRegression(random_state=1907)
        scorings = {**scoring, **prob_scoring}
    elif classifier == "etc":
        params = etc_params
        model = ExtraTreesClassifier(n_jobs=-1, random_state=1907)
        scorings = {**scoring, **prob_scoring}
    elif classifier == "adaboost":
        params = adaboost_params
        model = AdaBoostClassifier(random_state=1994)
        scorings = {**scoring, **prob_scoring}
    elif classifier == "gradboost":
        params = gradboost_params
        model = GradientBoostingClassifier(random_state=1994)
        scorings = {**scoring, **prob_scoring}
    else:
        raise Exception("Classifier {} is unknown!".format(classifier))
    return grid_search(model, params, scorings)

def main(argv):
    """
    The main function of the script.

    Flags:
        --tokenize          : Apply tokenizing and stemming.
        --load-tokenized    : Load existing tokenized and stemmed data.
        --load-lda          : Load existing trained LDA (with --num-topics X topics).
        --load-features     : Load saved features (of type --topic/term/char-features or raw).
        --filter-extremes   : Remove extreme tokens from corpus.
        --filter-types      : Remove only MBTI tokens.
        --topic-features    : Use topic distribution for document as feature vector.
        --term-features     : Use term topics as feature vector, together with TF.
        --char-features     : Use char frequency as feature vector.
        --num-topics X      : X: int, follow after --num-topics arg with space between.
        --normalize         :

    TODO: Write more here.
    """
    setup_logging()

    mbti = {
        'I':'Introversion', 'E':'Extroversion',
        'N':'Intuition', 'S':'Sensing',
        'T':'Thinking', 'F': 'Feeling',
        'J':'Judging', 'P': 'Perceiving'}

    if not "--load-tokenized" in argv:
        data_set = get_data(argv)
        types = sorted(set(data_set["type"]))
        #create_wordclouds(train)

        data_set["posts"].apply(handle_delimiter)

        data_set["posts"] = [post.lower() for post in data_set["posts"]]

        if "--tokenize" in argv or "-t" in argv:

            if "--filter-extremes" in argv:
                filter_level = 'extremes'
            elif "--filter-types" in argv:
                filter_level = 'types'
            else:
                filter_level = None

            data_set["posts"] = tokenize_and_stem(
                data_set["posts"],
                types,
                create_corpus=True,
                filter_level=filter_level)

            logging.info("Saving tokenized and stemmed data.")
            data_set.to_csv(TOKENIZED_DATA_FILE, index=False)
    else:
        logging.info("Loading tokenized data.")
        data_set = load_csv(TOKENIZED_DATA_FILE)
        data_set["posts"] = [str(post) for post in data_set["posts"]]
        types = sorted(set(data_set["type"]))

    scoring = {
        'acc': 'accuracy',
        'prec_micro': 'precision_micro',
        'rec_micro': 'recall_micro',
        'f1_micro': 'f1_micro'
    }

    prob_scoring = {
        'neg_log_loss': 'neg_log_loss'
    }

    X_train, X_test, y_train, y_test = train_test_split(
        data_set["posts"],
        data_set["type"],
        test_size=0.3,
        stratify=data_set["type"],
        random_state=1773)

    print(Counter(y_train), len(y_train))

    corpus = corpora.MmCorpus(CORPUS_FILE)
    dictionary = corpora.Dictionary.load(DICTIONARY_FILE)

    if "--load-lda" in argv:
        logging.info("Loading LDA for %s topics...", NUM_TOPICS)
        lda = LdaMulticore.load(LDA_FOLDER + "lda_model_{}".format(NUM_TOPICS))
    else:
        logging.info("Generating LDA for %s topics...", NUM_TOPICS)
        lda = LdaMulticore(
            corpus,
            num_topics=NUM_TOPICS,
            id2word=dictionary,
            workers=3,
            passes=50,
            batch=True,
            iterations=500)
        lda.save(LDA_FOLDER + "lda_model_{}".format(NUM_TOPICS))

    #print(lda.print_topics(num_topics=NUM_TOPICS, num_words=30))

    if "--term-features" in argv:
        feature_type = "terms"
    elif "--topic-features" in argv:
        feature_type = "topics"
    elif "--char-features" in argv:
        feature_type = "chars"
    else:
        logging.warning("Default features used (TF-IDF, etc.)")
        feature_type = "raw"

    if not "--load-features" in argv:
        logging.info("Extracting X_train features...")
        X_train_features, save_name = get_features(X_train, feature_type, lda, dictionary)
        np.save(X_TRAIN_FOLDER + "X_train_features_{}.npy".format(save_name), X_train_features)

        logging.info("Extracting X_test features...")
        X_test_features, save_name = get_features(X_test, feature_type, lda, dictionary)
        np.save(X_TEST_FOLDER + "X_test_features_{}.npy".format(save_name), X_test_features)
    else:
        logging.info("Loading feature vectors...")
        if feature_type == "chars" or feature_type == "raw":
            file_ending = feature_type
        elif feature_type == "topics" or feature_type == "terms":
            file_ending = "{}_{}".format(feature_type, NUM_TOPICS)
        else:
            raise Exception("Feature type {} is unknown!".format(feature_type))
        X_train_features = np.load(X_TRAIN_FOLDER + "X_train_features_{}.npy".format(file_ending))
        X_test_features = np.load(X_TEST_FOLDER + "X_test_features_{}.npy".format(file_ending))
    
    if "--normalize" in argv:
        X_train_features = normalize(X_train_features)
        X_test_features = normalize(X_test_features)

    #y_train = [mbti_type[2:4] for mbti_type in y_train]
    #y_test = [mbti_type[2:4] for mbti_type in y_test]

    if feature_type == "raw":
        #"extra_trees" "sgd"
        clf = get_model("etc", argv, X_train_features, y_train, scoring, prob_scoring)
    elif feature_type == "terms":
        clf = get_terms_classifier("logit", scoring, prob_scoring)
    elif feature_type == "topics":
        clf = get_topics_classifier("gradboost", scoring, prob_scoring)
    elif feature_type == "chars":
        clf = get_chars_classifier("linear", scoring, prob_scoring)
    else:
        raise Exception("Feature type {} is unknown".format(feature_type))

    #clf.fit(X_train_features, y_train)
    #logging.info("Best: %s, %s, %s", clf.best_index_, clf.best_score_, clf.best_params_)
    #clf = get_model("lr", argv, X_train, y_train, scoring) # extra_trees sgd lr
    logging.info("Testing clf: %s", clf)
    logging.info("Test set score: %s", clf.score(X_test_features, y_test))
    y_pred = clf.predict(X_test_features)
    logging.info(precision_recall_fscore_support(y_test, y_pred, average="micro"))
    print(Counter(y_pred))
    logging.info(classification_report(y_test, y_pred))
    

    my_own_data = [
        "I'm eagerly waiting for the next development on Social Media Platforms: being able to like likes on your/others' Social Media posts.",
        "The autumn semester was beyond all of my expectations. I've learned a lot of new things and gained new friends from all over the world. A big thank you to all of you for making my time here amazing. I'll never forget you. I hope the spring semester at ETH will be as memorable as the autumn one. <URL>",
        "Month of finals: 6 out of 7 exams done so  far with varying performance and 6 seasons of series completed. It's all about balance in life!",
        "Why do I always eat food while cooking? I'm always full by the time the dish is finished!",
        "Reliving my childhood #harrypotter #game <URL>",
        "Peridot loves Steven #stevenuniverse #pfefferkuchen #pepparkakor #gingerbreadcookies #selfmade #happyholidays <URL>",
        "Guess who's back on twitter? - this neeerd",
        "Java Lecture: When you feel like taking a nap.",
        "A bunch of friends on a friday night playin' #PropHunt! Awesome!",
        "Interesting day :D",
        "#dhopen @QuanticHyuN GG! you're the best!",
        "I stand by TotalBiscuit and the Terran Republic in the PlanetSide 2 Ultimate Showdown!  #PS2showdown",
        "longing for P tutorials with @ApolloSC2. in the mean time I'll ladder against High Gold/Diamond players! Thanks to you I'm now in gold!! :D",
        "Loosing hard in #SC2, MMMVG and Broodlords are really hard to deal with :(",
        "I uploaded a @YouTube video <URL>  [CoD: WaW] Nostalgia! Quick Match! 30 - 2",
        "I uploaded a @YouTube video <URL>  BF3 Test Footage",
        "Thanks! Now I know! <URL>  #ComputerPowerTest",
        "One does not simply make games without passion.",
        "I uploaded a @YouTube video <URL> Swetrox- - MW3 Game Clip",
        "I uploaded a @YouTube video <URL>  Shatterhand Audio School Project",
        "Finally some spare time! Time for #BF3!!! :D",
        "I uploaded a @YouTube video <URL>  Swetrox- - MW3 Game Clip",
        "I nominate @totalbiscuit for a Shorty Award in #gaming because he delivers entertaining top-quality gaming videos. <URL>",
        "#MW3 released a new game mode for FREE, and MIGHT release some free DLCs! What's that, #BF3? Right, you guys already do that!! :D",
        "Our first duet coming up soon! #LAN and #Singstar <3 The song: The Killers, When we were young!",
        "Land of Confusion! #Singstar",
        "I forgot: I also bought some APELSIN KROKANT! :D",
        "Bought some pizza, 1 Grape Tonic, 1 Grappo and 2 Ciders! #LAN",
        "We're up and running! #LAN time!! Gonna warm up with some #BF3 Wanna join? :D",
        "Would be awesome if I had any spare time to work on my #XNA game! Maybe do some bugfixing or animating the player? :D",
        "Enthusiastic about tomorrow's 18 hour LAN-Party! Gonna play soo many games! #StarCraft2  #BF3 #Sanctum being a few of 'em!",
        "#SC2 Time!",
        "Time to sleep. Tomorrow is a new day, filled with #Skyrim #StarCraft2 #XNA and #Floorball Good night! :D",
        "XNA Time!",
        "Just picked up mw3 :D"]
    my_own_data = [" ".join(my_own_data)]
    my_own_data = [post.lower() for post in my_own_data]

    if feature_type != "raw":
        my_own_data = tokenize_and_stem(my_own_data, types)
        my_own_data = [" ".join(my_own_data)]

    logging.info("Extracting features from my own data...")
    my_own_data_features = get_features(my_own_data, feature_type, lda, dictionary)[0]
    if "--normalize" in argv:
        my_own_data_features = normalize(my_own_data_features)
    print(my_own_data_features)
    print(clf.predict(my_own_data_features))
    try:
        predicted_classes = clf.predict_proba(my_own_data_features)
        pprint(list(zip(types, predicted_classes[0])))
    except Exception:
        print("Predict probabilities not supported...")
    return 0
    
    # def generate_learning_curve(model, X_train, y_train, cross_validation):
    #     train_sizes, train_scores, test_scores = learning_curve(
    #         model, 
    #         X_train, 
    #         y_train, 
    #         cv=cross_validation, 
    #         scoring="f1_micro", 
    #         train_sizes=np.linspace(.1, 1.0, 10), 
    #         random_state=1)
    
    #   plot_learning_curve(train['posts'], train['type'], train_sizes, 
    #                 train_scores, test_scores, ylim=(0.1, 1.01), figsize=(14,6))
    #   plt.show()

    #plot_types(data.type)
    #plot_subtypes(data.type)

    #print(data.sample(10))
    
    #posts = preprocess_posts(data.posts)
    #plot_media_link_types(posts)
    #posts = process_media_links(posts)




if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
    
    SAVED_DATA_FOLDER = "saved_data/"

    if "--filter-extremes" in sys.argv: 
        SAVED_DATA_FOLDER += "no_extremes/"
    elif "--filter-types" in sys.argv:
        SAVED_DATA_FOLDER += "no_types/"
    else:
        SAVED_DATA_FOLDER += "raw/"

    DATA_FILE = 'mbti_data_set.csv'
    TOKENIZED_DATA_FILE = SAVED_DATA_FOLDER + 'mbti_data_set_processed.csv'
    DICTIONARY_FILE = SAVED_DATA_FOLDER + 'tokens.dict'
    CORPUS_FILE = SAVED_DATA_FOLDER + 'corpus.mm'
    X_TRAIN_FOLDER = SAVED_DATA_FOLDER + "X_train/"
    X_TEST_FOLDER = SAVED_DATA_FOLDER + "X_test/"
    LDA_FOLDER = SAVED_DATA_FOLDER + "LDA/"

    if "--num-topics" in sys.argv:
        NUM_TOPICS = int(sys.argv[sys.argv.index("--num-topics") + 1])
    else:
        NUM_TOPICS = 100 

    main(sys.argv)
