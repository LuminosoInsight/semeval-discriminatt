import pkg_resources
import os

import attr
import numpy as np


# Use attr (an upgrade to namedtuple) to make a class that represents an
# example from the training/validation/test files.
AttributeExample = attr.make_class(
    'AttributeExample',
    ['word1', 'word2', 'attribute', 'discriminative']
)


def get_data_filename(filename):
    """
    Get a valid path referring to a given filename in the `data`
    subdirectory of the package.
    """
    return pkg_resources.resource_filename(
        'discriminatt', os.path.join('semeval-data', filename)
    )


def read_data(name):
    """
    Read the list of examples from one of the included data files.

    Example: read_data('training/train.txt')
    """
    filename = get_data_filename(name)
    examples = []
    for line in open(filename, encoding='utf-8'):
        word1, word2, attribute, strval = line.rstrip().split(',')
        discriminative = bool(int(strval))
        examples.append(AttributeExample(word1, word2, attribute, discriminative))
    return examples
