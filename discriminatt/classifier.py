import os
import sqlite3

import numpy as np
from conceptnet5.vectors.query import VectorSpaceWrapper, normalize_vec
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import normalize
from tqdm import tqdm as progress_bar

from discriminatt.data import read_semeval_data, get_external_data_filename, get_result_filename, read_phrases, \
    read_search_queries
from discriminatt.standalone_sme import StandaloneSMEModel
from discriminatt.wikipedia import wikipedia_connected_conceptnet_nodes
from discriminatt.wordnet import wordnet_connected_conceptnet_nodes


class AttributeClassifier:
    """
    Subclasses of this class are strategies for solving the task.

    Subclasses should override the .train(examples) and .classify(examples)
    methods.
    """
    def train(self, examples):
        """
        Given training examples (a list of AttributeExample objects),
        update the state of this classifier to learn from them.
        """
        pass

    def classify(self, examples):
        """
        Given test cases (a list of AttributeExample objects), return the
        classification of those test cases as a list or array of bools.

        Of course, the classification should only use the .text1, .text2,
        and .attribute properties of each example -- not the .discriminative
        property, which contains the answer.
        """
        raise NotImplementedError

    def evaluate(self):
        """
        Train this learning strategy, and evaluate its accuracy on the validation set.
        """
        training_examples = read_semeval_data('training/train.txt')
        test_examples = read_semeval_data('training/validation.txt')

        print("Training")
        self.train(training_examples)
        print("Testing")
        our_answers = np.array(self.classify(test_examples))
        real_answers = np.array([example.discriminative for example in test_examples])
        acc = np.equal(our_answers, real_answers).sum() / len(real_answers)
        return acc


class MultipleFeaturesClassifier(AttributeClassifier):
    """
    Compute a number of numeric features from the examples, based on different
    data sources. Then use the concatenation of all these features as input to
    an SVM.

    The values of each feature are cached in the `discriminatt/results`
    directory. If you change the code of a feature, delete its corresponding cache
    files from that directory.
    """
    def __init__(self):
        self.wrap = VectorSpaceWrapper(get_external_data_filename('numberbatch-20180108-biased.h5'),
                                       use_db=False)
        self.cache = {}
        self.wp_db = None
        self.sme = None
        self.queries = None
        self.phrases = None
        self.svm = None

        self.feature_methods = [
            self.direct_relatedness_features,
            self.wikipedia_relatedness_features,
            self.wordnet_relatedness_features,
            self.sme_features,
            self.search_query_features,
            self.phrase_hit_features
        ]

    def get_vector(self, uri):
        if uri in self.cache:
            return self.cache[uri]
        else:
            vec = normalize_vec(self.wrap.get_vector(uri))
            self.cache[uri] = vec
            return vec

    def get_similarity(self, uri1, uri2):
        return self.get_vector(uri1).dot(self.get_vector(uri2))

    def direct_relatedness_features(self, example):
        match1 = max(self.get_similarity(example.node1(), example.att_node()), 0) ** 0.5
        match2 = max(self.get_similarity(example.node2(), example.att_node()), 0) ** 0.5
        return np.array([match1 - match2])

    def wikipedia_relatedness_features(self, example):
        if self.wp_db is None:
            self.wp_db = sqlite3.connect(get_external_data_filename('wikipedia-summary.db'))
        connected1 = [example.node1()] + wikipedia_connected_conceptnet_nodes(self.wp_db, example.word1)
        connected2 = [example.node2()] + wikipedia_connected_conceptnet_nodes(self.wp_db, example.word2)
        return self.max_relatedness_features(connected1, connected2, example.att_node())

    def wordnet_relatedness_features(self, example):
        connected1 = [example.node1()] + wordnet_connected_conceptnet_nodes(example.word1)
        connected2 = [example.node2()] + wordnet_connected_conceptnet_nodes(example.word2)
        return self.max_relatedness_features(connected1, connected2, example.att_node())

    def max_relatedness_features(self, conn1, conn2, att_node):
        match1 = max([self.get_similarity(c, att_node) for c in conn1])
        match2 = max([self.get_similarity(c, att_node) for c in conn2])
        return np.array([match1 - match2])

    def sme_features(self, example):
        if self.sme is None:
            self.sme = StandaloneSMEModel(get_external_data_filename('sme-20171220'))
        features = []
        node1 = example.node1()
        node2 = example.node2()
        att = example.att_node()
        if node1 in self.sme and node2 in self.sme and att in self.sme:
            return self.sme.predict_discriminative_relations(node1, att) - self.sme.predict_discriminative_relations(node2, att)
        else:
            return np.zeros(self.sme.num_rels())

    def phrase_hit_features(self, example):
        if self.phrases is None:
            self.phrases = read_phrases('google-books-2grams.txt')
        word1_phrases = self.phrases[example.word1]
        word2_phrases = self.phrases[example.word2]
        att_phrases = self.phrases[example.attribute]
        int1 = set(word1_phrases).intersection(att_phrases)
        int2 = set(word2_phrases).intersection(att_phrases)
        if int1 and not int2:
            return np.array([1])
        else:
            return np.array([0])

    def search_query_features(self, example):
        if self.queries is None:
            self.queries = read_search_queries()
        word1_queries = self.queries[example.word1]
        word2_queries = self.queries[example.word2]
        att_queries = self.queries[example.attribute]
        int1 = set(word1_queries).intersection(att_queries)
        int2 = set(word2_queries).intersection(att_queries)
        difference = len(int1) - len(int2)
        if difference > 0:
            return [np.log(difference)]
        else:
            return [0]

    def extract_features(self, examples, mode='train'):
        subarrays = []
        for method in self.feature_methods:
            name = method.__name__
            feature_filename = get_result_filename('{}.{}.npy'.format(name, mode))
            try:
                os.mkdir(os.path.dirname(feature_filename))
            except FileExistsError:
                pass
            if os.access(feature_filename, os.R_OK):
                features = np.load(feature_filename)
            else:
                feature_list = []
                for example in progress_bar(examples, desc=name):
                    feature_list.append(method(example))
                features = np.vstack(feature_list)
                np.save(feature_filename, features)
            subarrays.append(features)
        return np.hstack(subarrays)

    def train(self, examples):
        self.svm = LinearSVC()
        inputs = normalize(self.extract_features(examples, mode='train'), axis=0, norm='l2')
        outputs = np.array([example.discriminative for example in examples])
        self.svm.fit(inputs, outputs)
        print(self.svm.coef_)

    def classify(self, examples):
        inputs = normalize(self.extract_features(examples, mode='test'), axis=0, norm='l2')
        predictions = self.svm.predict(inputs)
        return predictions


if __name__ == '__main__':
    multiple_features = MultipleFeaturesClassifier()
    print(multiple_features.evaluate())
