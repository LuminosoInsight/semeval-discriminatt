import pkg_resources
import os

from attr import attrs, attrib
import numpy as np
from conceptnet5.nodes import concept_uri


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
        return concept_uri('en', self.word1)

    def node2(self):
        "Get word2 as a ConceptNet URI."
        return concept_uri('en', self.word2)

    def att_node(self):
        "Get the attribute as a ConceptNet URI."
        return concept_uri('en', self.attribute)


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


def read_phrases(name):
    """
     TODO
    """
    filename = get_external_data_filename(name)
    with open(filename, encoding='utf-8') as input_file:
        data = set(line.split(',')[0].lower() for line in input_file)
    return data
