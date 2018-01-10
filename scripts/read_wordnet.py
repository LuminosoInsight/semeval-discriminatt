import nltk
import json
from nltk.corpus import wordnet

get_synset = wordnet._synset_from_pos_and_offset


def get_adjacent(synset):
    return [
        name
        for pointer_tuples in synset._pointers.values()
        for pos, offset in pointer_tuples
        for name in get_synset(pos, offset).lemma_names()
    ]


def iter_wordnet_entries():
    synsets = wordnet.all_synsets()
    for syn in synsets:
        lemmas = [lem.replace('_', ' ') for lem in syn.lemma_names()]
        related = [lem.replace('_', ' ') for lem in get_adjacent(syn)]

        output = {
            'lemmas': lemmas,
            'related': related,
            'definition': syn.definition(),
            'examples': syn.examples()
        }
        yield output


def run():
    for entry in iter_wordnet_entries():
        print(json.dumps(entry, ensure_ascii=False))


if __name__ == '__main__':
    run()

