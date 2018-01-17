import numpy as np
from tqdm import tqdm as progress_bar
import sqlite3
import os
from sklearn.svm import SVC

from conceptnet5.vectors.query import VectorSpaceWrapper, normalize_vec
from conceptnet5.nodes import concept_uri
from discriminatt.data import AttributeExample, read_semeval_data, get_external_data_filename, get_result_filename, read_phrases
from discriminatt.wordnet import wordnet_connected_conceptnet_nodes
from discriminatt.wikipedia import wikipedia_connected_conceptnet_nodes


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
        acc = np.equal(our_answers, real_answers) / len(real_answers)
        return acc


class MultipleFeaturesClassifier(AttributeClassifier):
    def __init__(self, embeddings_filename, phrases_filename, wikipedia_filename):
        self.wrap = VectorSpaceWrapper(get_external_data_filename(embeddings_filename), use_db=False)
        self.cache = {}
        self.phrases = read_phrases(phrases_filename)
        self.wp_db = sqlite3.connect(get_external_data_filename(wikipedia_filename))
        self.svm = None

        self.feature_methods = [
            self.direct_relatedness_features,
            self.wikipedia_relatedness_features,
            self.wordnet_relatedness_features,
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
        term1 = concept_uri('en', example.word1)
        term2 = concept_uri('en', example.word2)
        att = concept_uri('en', example.attribute)
        match1 = self.get_similarity(term1, att)
        match2 = self.get_similarity(term2, att)
        return np.array([match1, match2])

    def wikipedia_relatedness_features(self, example):
        term1 = concept_uri('en', example.word1)
        term2 = concept_uri('en', example.word2)
        connected1 = [term1] + wikipedia_connected_conceptnet_nodes(self.wp_db, example.word1)
        connected2 = [term2] + wikipedia_connected_conceptnet_nodes(self.wp_db, example.word2)
        return self.max_relatedness_features(connected1, connected2, example.attribute)

    def wordnet_relatedness_features(self, example):
        term1 = concept_uri('en', example.word1)
        term2 = concept_uri('en', example.word2)
        connected1 = [term1] + wordnet_connected_conceptnet_nodes(example.word1)
        connected2 = [term2] + wordnet_connected_conceptnet_nodes(example.word2)
        return self.max_relatedness_features(connected1, connected2, example.attribute)

    def max_relatedness_features(self, conn1, conn2, attribute):
        att = concept_uri('en', attribute)
        match1 = max([self.get_similarity(c, att) for c in conn1])
        match2 = max([self.get_similarity(c, att) for c in conn2])
        return np.array([match1, match2])

    def phrase_hit_features(self, example):
        phrase1 = '{} {}'.format(example.word1, example.attribute)
        phrase2 = '{} {}'.format(example.word2, example.attribute)
        return np.array([int(phrase1 in self.phrases), int(phrase2 in self.phrases)])

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
        self.svm = SVC()
        inputs = self.extract_features(examples, mode='train')
        outputs = np.array([example.discriminative for example in examples])
        self.svm.fit(inputs, outputs)

    def classify(self, examples):
        inputs = self.extract_features(examples, mode='test')
        predictions = self.svm.predict(inputs)
        return predictions


if __name__ == '__main__':
    multiple_features = MultipleFeaturesClassifier(
        'numberbatch-20180108-biased.h5',
        'google-books-2grams.txt',
        'wikipedia-summary.db'
    )
    print(multiple_features.evaluate())
