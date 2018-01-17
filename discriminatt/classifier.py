import numpy as np
from tqdm import tqdm as progress_bar
import sqlite3
from sklearn.svm import SVC

from conceptnet5.vectors.query import VectorSpaceWrapper, normalize_vec
from conceptnet5.nodes import concept_uri
from discriminatt.data import AttributeExample, read_semeval_data, get_external_data_filename, read_phrases
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

        self.train(training_examples)
        our_answers = np.array(self.classify(test_examples))
        real_answers = np.array([example.discriminative for example in test_examples])

        acc = (our_answers == real_answers).sum() / len(real_answers)
        return acc


class ConstantBaselineClassifier(AttributeClassifier):
    """
    A trivial example of a classification strategy.

    It gets 50.1% accuracy on the validation data, because that data has
    roughly equal numbers of true and false examples.
    """
    def classify(self, examples):
        return np.array([True] * len(examples))


class RelatednessClassifier(AttributeClassifier):
    """
    A straightforward but under-powered strategy that uses word vectors.

    Compare the relatedness of (word1, attribute) and the relatedness of
    (word2, attribute), using the provided VectorSpaceWrapper of word vectors.
    The two relatedness scores are given to an SVM classifier, which decides
    how these scores determine whether the attribute is discriminative.

    Scores 56.5% using a recent ConceptNet Numberbatch mini.h5.
    """
    def __init__(self, embedding_filename):
        self.wrap = VectorSpaceWrapper(embedding_filename, use_db=False)
        self.svm = None

    def find_relatedness(self, examples, desc='Training'):
        relatedness_by_example = []
        for example in progress_bar(examples, desc=desc):
            term1 = concept_uri('en', example.word1)
            term2 = concept_uri('en', example.word2)
            att = concept_uri('en', example.attribute)

            match1 = self.wrap.get_similarity(term1, att)
            match2 = self.wrap.get_similarity(term2, att)
            connected_match1 = max([
                self.wrap.get_similarity(c, att)
                for c in wordnet_connected_conceptnet_nodes(example.word1)
            ])
            connected_match2 = max([
                self.wrap.get_similarity(c, att)
                for c in wordnet_connected_conceptnet_nodes(example.word2)
            ])
            relatedness_by_example.append([match1, match2, connected_match1, connected_match2])
        return relatedness_by_example

    def train(self, examples):
        self.svm = SVC()
        inputs = self.find_relatedness(examples, desc='Training')
        outputs = np.array([example.discriminative for example in examples])
        self.svm.fit(inputs, outputs)

    def classify(self, examples):
        inputs = self.find_relatedness(examples, desc='Testing')
        predictions = self.svm.predict(inputs)
        return predictions


class MultipleFeaturesClassifier(AttributeClassifier):
    def __init__(self, embeddings_filename, phrases_filename, wikipedia_filename):
        self.wrap = VectorSpaceWrapper(get_external_data_filename(embeddings_filename), use_db=False)
        self.cache = {}
        self.phrases = read_phrases(phrases_filename)
        self.wp_db = sqlite3.connect(get_external_data_filename(wikipedia_filename))
        self.svm = None

    def get_vector(self, uri):
        if uri in self.cache:
            return self.cache[uri]
        else:
            vec = normalize_vec(get_vector(self.wrap.frame, uri))
            self.cache[uri] = vec
            return vec

    def get_similarity(self, uri1, uri2):
        return self.get_vector(uri1).dot(self.get_vector(uri2))

    def find_relatedness(self, example):
        relatedness_by_example = []
        term1 = concept_uri('en', example.word1)
        term2 = concept_uri('en', example.word2)
        att = concept_uri('en', example.attribute)

        match1 = self.wrap.get_similarity(term1, att)
        match2 = self.wrap.get_similarity(term2, att)
        connected_match1 = max([
            self.wrap.get_similarity(c, att)
            for c in self.connected_nodes(example.word1)
        ])
        connected_match2 = max([
            self.wrap.get_similarity(c, att)
            for c in self.connected_nodes(example.word2)
        ])
        return [match1, match2, connected_match1, connected_match2]

    def find_phrase_hit(self, example):
        phrase1 = '{} {}'.format(example.word1, example.attribute)
        phrase2 = '{} {}'.format(example.word2, example.attribute)
        return [phrase1 in self.phrases, phrase2 in self.phrases]

    def connected_nodes(self, word):
        wordnet_nodes = wordnet_connected_conceptnet_nodes(word)
        wp_nodes = wikipedia_connected_conceptnet_nodes(self.wp_db, word)
        connected_nodes = wordnet_nodes + wp_nodes
        return connected_nodes

    def extract_features(self, examples, desc):
        features = []
        for example in progress_bar(examples, desc=desc):
            relatedness = self.find_relatedness(example)
            phrase_hits = self.find_phrase_hit(example)
            features.append(relatedness + phrase_hits)
        return features

    def train(self, examples):
        self.svm = SVC()
        inputs = self.extract_features(examples, desc='Training')
        outputs = np.array([example.discriminative for example in examples])
        self.svm.fit(inputs, outputs)

    def classify(self, examples):
        inputs = self.extract_features(examples, desc='Testing')
        predictions = self.svm.predict(inputs)
        return predictions


if __name__ == '__main__':
    cl = ConstantBaselineClassifier()
    print(cl.evaluate())

    multiple_features = MultipleFeaturesClassifier(
        'numberbatch-20180108-biased.h5',
        'google-books-2grams.txt',
        'wikipedia.db'
    )
    print(multiple_features.evaluate())
