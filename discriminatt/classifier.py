import os
import sqlite3
import itertools

import numpy as np
import pandas as pd
from conceptnet5.vectors.query import VectorSpaceWrapper, normalize_vec
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from tqdm import tqdm as progress_bar

from discriminatt.data import (
    read_semeval_data, read_blind_semeval_data,
    get_external_data_filename, get_result_filename, read_search_queries
)
from discriminatt.phrases import phrase_weight
from discriminatt.standalone_sme import StandaloneSMEModel
from discriminatt.wikipedia import wikipedia_connected_conceptnet_nodes
from discriminatt.wordnet import wordnet_connected_conceptnet_nodes


np.random.seed(0)


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

    def show_acc(self, our_answers, real_answers, mode):
        acc = np.equal(our_answers, real_answers).sum() / len(real_answers)
        acc_error = ((acc * (1 - acc)) / len(real_answers)) ** 0.5
        print("{} accuracy: {:.2%} ± {:.2%}".format(mode, acc, acc_error))
        return acc

    def evaluate(self):
        """
        Train this learning strategy, and evaluate its accuracy on the validation set.
        """
        examples = {
            'train': read_semeval_data('training/train.txt'),
            'validation': read_semeval_data('training/validation.txt'),
            'test': read_semeval_data('test/truth.txt')
        }

        self.train(examples['train'])
        accuracies = {}
        for mode in ['train', 'validation', 'test']:
            this_acc = self.show_acc(
                self.classify(examples[mode], mode),
                [example.discriminative for example in examples[mode]],
                mode
            )
            accuracies[mode] = this_acc
        return accuracies

    def run_test(self):
        test_examples = read_blind_semeval_data('test/test_triples.txt')
        output = open(get_result_filename('answer.txt'), 'w')
        our_answers = np.array(self.classify(test_examples, 'test'))
        for example, answer in zip(test_examples, our_answers):
            print(
                "{},{},{},{}".format(
                    example.word1, example.word2, example.attribute, int(answer)
                ),
                file=output
            )


class MultipleFeaturesClassifier(AttributeClassifier):
    """
    Compute a number of numeric features from the examples, based on different
    data sources. Then use the concatenation of all these features as input to
    an SVM.

    The values of each feature are cached in the `discriminatt/results`
    directory. If you change the code of a feature, delete its corresponding cache
    files from that directory.
    """
    def __init__(self, ablate=()):
        self.wrap = VectorSpaceWrapper(get_external_data_filename('numberbatch-20180108-biased.h5'),
                                       use_db=False)
        self.cache = {}
        self.wp_db = None
        self.sme = None
        self.queries = None
        self.phrases = None
        self.svm = None
        self.ablate = ablate

        self.feature_methods = [
            self.direct_relatedness_features,
            self.sme_features,
            self.wikipedia_relatedness_features,
            self.wordnet_relatedness_features,
            self.phrase_hit_features
        ]

        self.feature_names = [
            'ConceptNet vector relatedness',
            'SME: RelatedTo',
            'SME: (x IsA a)',
            'SME: (x HasA a)',
            'SME: (x PartOf a)',
            'SME: (x CapableOf a)',
            'SME: (x UsedFor a)',
            'SME: (x HasContext a)',
            'SME: (x HasProperty a)',
            'SME: (x AtLocation a)',
            'SME: (a PartOf x)',
            'SME: (a AtLocation x)',
            'Wikipedia lead sections',
            'WordNet relatedness',
            'Google Ngrams',
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
            self.sme = StandaloneSMEModel(get_external_data_filename('sme-20180129'))
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
            self.phrases = sqlite3.connect(get_external_data_filename('phrases.db'))
        weight_pair1 = phrase_weight(self.phrases, example.lemma1(), example.lemma_att())
        weight_pair2 = phrase_weight(self.phrases, example.lemma2(), example.lemma_att())
        return weight_pair1 - weight_pair2

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
            return np.log(difference)
        else:
            return 0

    def extract_features(self, examples, mode='train'):
        subarrays = []
        for i, method in enumerate(self.feature_methods):
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

            # Set a selected feature source to all zeroes
            if i in self.ablate:
                features *= 0
            subarrays.append(features)
        return np.hstack(subarrays)

    def train(self, examples):
        self.svm = LinearSVC()
        inputs = normalize(self.extract_features(examples, mode='train'), axis=0, norm='l2')
        outputs = np.array([example.discriminative for example in examples])
        self.svm.fit(inputs, outputs)

        # Zero out features that get a negative weight -- these features were
        # intended to be positive, so one that comes out negative is probably
        # overfitting
        self.svm.coef_ = np.maximum(0, self.svm.coef_)
        coef_series = pd.Series(self.svm.coef_[0], index=self.feature_names)
        if self.ablate:
            used_feature_names = [
                self.feature_methods[a].__name__
                for a in range(5)
                if a not in self.ablate
            ]
            print("Used [{}]".format(', '.join(used_feature_names)))
        else:
            print(coef_series)

    def classify(self, examples, mode):
        inputs = normalize(self.extract_features(examples, mode=mode), axis=0, norm='l2')
        predictions = self.svm.predict(inputs)
        return predictions


if __name__ == '__main__':
    multiple_features = MultipleFeaturesClassifier()
    print(multiple_features.evaluate())
    multiple_features.run_test()

    labels = []
    valid_accs = []
    test_accs = []

    for n_drop in range(5):
        for ablation in itertools.combinations(range(5), r=n_drop):
            short_id = ''.join(ch for ch in 'ABCDE' if (ord(ch) - ord('A')) not in ablation)
            print()
            print(short_id)
            ablated = MultipleFeaturesClassifier(ablation)
            accuracies = ablated.evaluate()

            labels.append(short_id)
            valid_accs.append(accuracies['validation'])
            test_accs.append(accuracies['test'])

    labels.reverse()
    valid_accs.reverse()
    test_accs.reverse()

    print("labels = {}".format(labels))
    print("valid_accs = {}".format(valid_accs))
    print("test_accs = {}".format(test_accs))

