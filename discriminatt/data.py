import os
from collections import defaultdict

import pkg_resources
from attr import attrs, attrib
from conceptnet5.language.english import LEMMATIZER
from conceptnet5.nodes import standardized_concept_uri


# Use attr (an upgrade to namedtuple) to make a class that represents an
# example from the training/validation/test files.

@attrs
class AttributeExample:
    """
    A data class that holds an individual example from this SemEval task.
    """
    word1 = attrib()
    word2 = attrib()
    attribute = attrib()
    discriminative = attrib()

    def node1(self):
        "Get word1 as a ConceptNet URI."
        return standardized_concept_uri('en', self.word1)

    def node2(self):
        "Get word2 as a ConceptNet URI."
        return standardized_concept_uri('en', self.word2)

    def att_node(self):
        "Get the attribute as a ConceptNet URI."
        return standardized_concept_uri('en', self.attribute)

    def lemma1(self):
        return LEMMATIZER.lookup('en', self.word1)[0]

    def lemma2(self):
        return LEMMATIZER.lookup('en', self.word2)[0]

    def lemma_att(self):
        return LEMMATIZER.lookup('en', self.attribute)[0]


def get_semeval_data_filename(filename):
    """
    Get a valid path referring to a given filename in the `semeval-data`
    subdirectory of the package.
    """
    return pkg_resources.resource_filename(
        'discriminatt', os.path.join('semeval-data', filename)
    )


def get_external_data_filename(filename):
    """
    Get a valid path referring to a given filename in the `more-data`
    subdirectory of the package.
    """
    return pkg_resources.resource_filename(
        'discriminatt', os.path.join('more-data', filename)
    )


def get_result_filename(filename):
    """
    Get a valid path referring to a given filename in the `results`
    subdirectory of the package.
    """
    return pkg_resources.resource_filename(
        'discriminatt', os.path.join('results', filename)
    )


def read_semeval_data(name):
    """
    Read the list of examples from one of the included data files.

    Example: read_semeval_data('training/train.txt')
    """
    filename = get_semeval_data_filename(name)
    examples = []
    for line in open(filename, encoding='utf-8'):
        word1, word2, attribute, strval = line.rstrip().split(',')
        discriminative = bool(int(strval))
        examples.append(AttributeExample(word1, word2, attribute, discriminative))
    return examples


def read_semeval_blind_data(name):
    """
    Read the list of examples from the test file, which does not contain the
    correct answers.

    Example: read_semeval_data('test/test_triples.txt')
    """
    filename = get_semeval_data_filename(name)
    examples = []
    for line in open(filename, encoding='utf-8'):
        word1, word2, attribute = line.rstrip().split(',')
        examples.append(AttributeExample(word1, word2, attribute, None))
    return examples


def read_search_queries():
    """
    Read AOL Query Logs and construct an index mapping a word to each document in which it appeared.
    """
    queries_index = defaultdict(list)
    offset = 0
    for number in range(1, 11):
        filename = get_external_data_filename('user-ct-test-collection-{0:02d}.txt'.format(number))
        doc_number = 0
        with open(filename, encoding='utf-8') as input_file:
            for i, line in enumerate(input_file):
                if i == 0: # skip the header line
                    continue
                query = line.split('\t')[1]
                words = query.split()
                if len(words) == 1:
                    continue
                for word in words:
                    queries_index[word].append(i + offset)
                doc_number = i
            offset += doc_number
    return queries_index
