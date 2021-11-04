# imports
# your basics
import re
import numpy as np
import pandas as pd

# nltk: http://www.nltk.org/api/nltk.lm.html#module-nltk.lm 
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.lm.preprocessing import padded_everygram_pipeline 
from nltk.lm import MLE # train a Maximum Likelihood Estimator

# scikit learn: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics


# tokenizers

def tokenize_ideal(line): # rasmus
    tokens = []
    # unmatchables = []
    
    for word in line.split():
        if re.findall(r"\w+-\w+|\w+|[.&?%!#…]", word) != []:
            x = re.findall(r"\w+-\w+|\w+|[.&?%!#…]+", word)
            for element in x:
                tokens.append(element)

        # if re.findall(r"\w+-\w+|\w+|[.&?%!#…]", word) != [word] and re.findall(r"[^\w|.&!?%#…]+", word) != []:
        #     unmatchables.append(re.findall(r"[^\w|.!#?%…&]+", word)[0])


    return tokens#, unmatchables

# functions

def mle_perplexity(tweets, n):
    '''
    Returns a list with the perplexity score for the first 5 tweets in a list.

            Parameters:
                    tweets (dataframe): a list of lists containing strings.
                    n (int): n-grams. Use 2 for bi-grams.

            Returns:
                    perplexity_tests (list): list of floats.
    '''
    tokenized_train = []

    for tweet in tweets:
        tokenized_train.append(tokenize_ideal(tweet)[0])
    
    train, vocab = padded_everygram_pipeline(n, tokenized_train)

    lm = MLE(2) 
    lm.fit(train, vocab)

    perplexity_tests = []
    for i in range(6):
        test = list(bigrams(text[i]))
        perplexity_tests.append(lm.perplexity(test))

    return perplexity_tests



def tweet_to_list(path):
    df = pd.read_csv(path, sep="\t", quoting=3, names=["col"])
    the_list = []
    for value in df["col"]:
        the_list.append(value)
    return the_list

def label_to_list(path):
    return pd.read_csv(path, sep="\t", quoting=3, names=["labels"]).values.flatten()

def paths_to_list(folder):
    # train data
    train_text = '../datasets/' + folder + '/train_text.txt'
    train_labels = '../datasets/' + folder + '/train_labels.txt'

    # validation data
    val_text = '../datasets/' + folder + '/val_text.txt'
    val_labels = '../datasets/' + folder + '/val_labels.txt'

    # test data
    test_text = '../datasets/' + folder + '/test_text.txt'
    test_labels = '../datasets/' + folder + '/test_labels.txt'
    

    train_text = tweet_to_list(train_text)
    train_labels = label_to_list(train_labels)

    val_text = tweet_to_list(val_text)
    val_labels = label_to_list(val_labels)

    test_text = tweet_to_list(test_text)
    test_labels = label_to_list(test_labels)


    return train_text, train_labels, test_text, test_labels, val_text, val_labels

def SGD_accuracy(folder, loss):
    train_text, train_labels, test_text, test_labels, val_text, val_labels = paths_to_list(folder)

    text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss=loss)),
    ])

    text_clf.fit(train_text, train_labels)

    predicted = text_clf.predict(val_text)
    accuracy_val = np.mean(predicted == val_labels)

    predicted = text_clf.predict(test_text)
    accuracy_test = np.mean(predicted == test_labels)

    return accuracy_val, accuracy_test




print('Imports and functions loaded.')