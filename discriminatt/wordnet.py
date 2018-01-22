from conceptnet5.language.english import LEMMATIZER
from conceptnet5.nodes import standardized_concept_uri
from nltk.corpus import wordnet
from wordfreq.tokens import tokenize

get_synset = wordnet._synset_from_pos_and_offset


def get_adjacent(synset):
    return [
        name
        for pointer_tuples in synset._pointers.values()
        for pos, offset in pointer_tuples
        for name in get_synset(pos, offset).lemma_names()
    ]


def get_reasonable_synsets(word):
    lemmas = wordnet.lemmas(word)
    cnet_lemma, _pos = LEMMATIZER.lookup('en', word)
    if cnet_lemma != word:
        lemmas += wordnet.lemmas(cnet_lemma)
    good_synsets = []
    for lem in lemmas:
        syn = lem.synset()
        if syn.lemma_names()[0] == word:
            good_synsets.append(syn)
    if not good_synsets:
        return [lem.synset() for lem in lemmas]
    else:
        return good_synsets


def get_wordnet_entries(word):
    word = word.lower()
    synsets = get_reasonable_synsets(word)
    results = []
    for syn in synsets:
        related = [lem.replace('_', ' ') for lem in get_adjacent(syn)]
        info = {
            'related': related,
            'definition': syn.definition(),
            'examples': syn.examples()
        }
        results.append(info)
    return results


def get_wordnet_connected_words(word):
    words = []
    for entry in get_wordnet_entries(word):
        words.extend(tokenize(entry['definition'], 'en'))
        words.extend(entry['related'])
        for example in entry['examples']:
            words.extend(tokenize(example, 'en'))
    return words


def wordnet_connected_conceptnet_nodes(word):
    nodes = [standardized_concept_uri('en', word)]
    for conn in get_wordnet_connected_words(word):
        nodes.append(standardized_concept_uri('en', conn))
    return nodes

