import logging
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict
from os import listdir
from os.path import splitext
import time
import re
import warnings

from wordcloud import WordCloud
from bs4 import BeautifulSoup
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier

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

def filename_timestamp(name):
    return name + "_%02d%02d%02d%02d%02d" % (TIME.tm_mon, TIME.tm_mday, TIME.tm_hour, TIME.tm_min, TIME.tm_sec)

def saveresult(result):
    filename = filename_timestamp("result")
    with open("./" + filename + ".csv", "w") as f:
        f.write("ID,Prediction\n")
        i = 0
        for line in result:
            i += 1
            f.write("{},{}\n".format(i, line[1]))
    logging.info("Result saved as %s", filename)


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

    generate_wordcloud(all_words, "wordcloud_all")

    for mbti_type, words in words_for_type.items():
        generate_wordcloud(words, "wordcloud_{}".format(mbti_type))
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
    Function that either loads the raw MBTI data set and preprocesses it (optionally also saving it)
    or loads an already saved (preprocessed) data set.
    Returns the processed data set.
    """
    # TODO: Give data file as command line argument
    if "--load" in argv or "-l" in argv: # Load saved data
        logging.info("Loading preprocessed data.")
        return load_csv(SAVED_DATA_FILE)

    logging.info("Loading raw data.")
    train = load_csv(DATA_FILE)

    if "--no-preprocess" in argv:
        logging.info("Using raw data without preprocessing.")
        return train

    logging.info("Using loaded, preprocessed data.")
    train["posts"] = preprocess_posts(train["posts"])

    if "--save" in argv or "-s" in argv: # Save data to file
        logging.info("Saving preprocessed raw data.")
        train.to_csv(SAVED_DATA_FILE, index=False)
    
    return train

def main(argv):
    """
    The main function of the script.

    Flags:
        -l (--load): Load an already saved (preprocessed) data set.
        -s (--save): Save the raw data set after preprocessing it.
        --no-preprocessing: Skip preprocessing of data (only if raw data set is used).

    TODO: Write more here.
    """
    setup_logging()
    #print(argv)

    mbti = {
        'I':'Introversion', 'E':'Extroversion',
        'N':'Intuition', 'S':'Sensing',
        'T':'Thinking', 'F': 'Feeling',
        'J':'Judging', 'P': 'Perceiving'}

    train = get_data(argv)

    #create_wordclouds(train)

    train["posts"].apply(handle_delimiter)

    X_train, X_test, y_train, y_test = train_test_split(
        train["posts"], train["type"], test_size=0.33, random_state=1)

    steps = [
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('tsvd', TruncatedSVD()),
        ('extra_trees', ExtraTreesClassifier())]
    
    pipeline = Pipeline(steps)
    logging.info("Pipeline set up: %s", pipeline)

    kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    logging.info("Using kfolds: %s", kfolds)

    scoring = {'AUC': 'roc_auc',
               'acc': 'accuracy',
               'neg_log_loss': 'neg_log_loss',
               'f1_micro': 'f1_micro'}

    parameters = {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tsvd__n_components': [10, 15, 20],
        'extra_trees__n_estimators': [10, 20, 30],
        'extra_trees__max_depth': [None, 3, 4, 5]
    }

    cv = GridSearchCV(pipeline, parameters, cv=kfolds, n_jobs=-1)
    cv.fit(X_train, y_train)
    y_predictions = cv.predict(X_test)
    report = sklearn.metrics.classification_report(y_test, y_predictions)
    print(report)
    #results = clf.cv_results_
    #print(results)

    #logging.info("CV Accuracy: {:0.4f} (+/- {:0.4f})".format(
    #    np.mean(results['test_acc']),
    #    np.std(results['test_acc']) * 2))

    #logging.info("CV F1: {:0.4f} (+/- {:0.4f})".format(
    #    np.mean(results['test_f1_micro']),
    #    np.std(results['test_f1_micro']) * 2))

    #logging.info("CV Logloss: {:0.4f} (+/- {:0.4f})".format(
    #    np.mean(-1*results['test_neg_log_loss']),
    #    np.std(-1*results['test_neg_log_loss']) * 2))

    return 0
    tfidf2 = CountVectorizer(ngram_range=(1, 1), 
                         stop_words='english',
                         lowercase = True, 
                         max_features = 5000)

    model_nb = Pipeline([('tfidf1', tfidf2), ('nb', MultinomialNB())])

    results_nb = cross_validate(model_nb, train['clean_posts'], train['type'], cv=kfolds, 
                            scoring=scoring)

    print("CV Accuracy: {:0.4f} (+/- {:0.4f})".format(np.mean(results_nb['test_acc']),
                                                          np.std(results_nb['test_acc'])))

    print("CV F1: {:0.4f} (+/- {:0.4f})".format(np.mean(results_nb['test_f1_micro']),
                                                          np.std(results_nb['test_f1_micro'])))

    print("CV Logloss: {:0.4f} (+/- {:0.4f})".format(np.mean(-1*results_nb['test_neg_log_loss']),
                                                          np.std(-1*results_nb['test_neg_log_loss'])))

    tfidf2 = CountVectorizer(ngram_range=(1, 1), stop_words='english',
                                                 lowercase = True, max_features = 5000)

    model_lr = Pipeline([('tfidf1', tfidf2), ('lr', LogisticRegression(class_weight="balanced", C=0.005))])

    results_lr = cross_validate(model_lr, train['clean_posts'], train['type'], cv=kfolds, 
                            scoring=scoring)

    print("CV Accuracy: {:0.4f} (+/- {:0.4f})".format(np.mean(results_lr['test_acc']),
                                                          np.std(results_lr['test_acc'])))

    print("CV F1: {:0.4f} (+/- {:0.4f})".format(np.mean(results_lr['test_f1_micro']),
                                                            np.std(results_lr['test_f1_micro'])))

    print("CV Logloss: {:0.4f} (+/- {:0.4f})".format(np.mean(-1*results_lr['test_neg_log_loss']),
                                                            np.std(-1*results_lr['test_neg_log_loss'])))
    
    train_sizes, train_scores, test_scores = \
    learning_curve(model_lr, train['clean_posts'], train['type'], cv=kfolds, 
                   scoring="f1_micro", train_sizes=np.linspace(.1, 1.0, 10), random_state=1)
    
    plot_learning_curve(train['posts'], train['type'], train_sizes, 
                    train_scores, test_scores, ylim=(0.1, 1.01), figsize=(14,6))
    plt.show()
    #plot_types(data.type)
    #plot_subtypes(data.type)

    #print(data.sample(10))
    
    #posts = preprocess_posts(data.posts)
    #plot_media_link_types(posts)
    #posts = process_media_links(posts)




if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

    DATA_FILE = 'mbti_data_set.csv'
    SAVED_DATA_FILE = 'mbti_data_set_processed.csv'
    TIME = time.localtime(time.time())

    main(sys.argv)
