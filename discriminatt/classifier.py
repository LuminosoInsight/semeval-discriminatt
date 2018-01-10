import numpy as np
from tqdm import tqdm as progress_bar
from sklearn.svm import SVC

from conceptnet5.vectors.query import VectorSpaceWrapper
from conceptnet5.nodes import concept_uri
from conceptnet5.util import get_data_filename as get_conceptnet_data_filename
from discriminatt.data import AttributeExample, read_data, read_phrases


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
        training_examples = read_data('training/train.txt')
        test_examples = read_data('training/validation.txt')

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
        self.wrap = VectorSpaceWrapper(embedding_filename)
        self.svm = None

    def find_relatedness(self, examples, desc='Training'):
        relatedness_by_example = []
        for example in progress_bar(examples, desc=desc):
            term1 = concept_uri('en', example.word1)
            term2 = concept_uri('en', example.word2)
            att = concept_uri('en', example.attribute)

            match1 = self.wrap.get_similarity(term1, att)
            match2 = self.wrap.get_similarity(term2, att)
            relatedness_by_example.append([match1, match2])
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
    def __init__(self, embeddings_filename, phrases_filename):
        self.wrap = VectorSpaceWrapper(embeddings_filename)
        self.phrases = read_phrases(phrases_filename)
        self.svm = None

    def find_relatedness(self, example):
        term1 = concept_uri('en', example.word1)
        term2 = concept_uri('en', example.word2)
        att = concept_uri('en', example.attribute)

        match1 = self.wrap.get_similarity(term1, att)
        match2 = self.wrap.get_similarity(term2, att)
        return [match1, match2]

    def find_phrase_hit(self, example):
        phrase1 = '{} {}'.format(example.word1, example.attribute)
        phrase2 = '{} {}'.format(example.word2, example.attribute)
        return [phrase1 in self.phrases, phrase2 in self.phrases]

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

    conceptnet_relatedness = RelatednessClassifier(get_conceptnet_data_filename('vectors-20180108/numberbatch-biased.h5'))
    print(conceptnet_relatedness.evaluate())

    multiple_features = MultipleFeaturesClassifier(get_conceptnet_data_filename(
        'vectors-20180108/numberbatch-biased.h5'), 'google-books-2grams.txt')
    print(multiple_features.evaluate())
