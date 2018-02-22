This is Luminoso's entry to SemEval-2018 task 10, "[Capturing Discriminative
Attributes](https://competitions.codalab.org/competitions/17326)".

It uses information from ConceptNet, WordNet, Wikipedia, and Google Ngrams as inputs to
a simple linear classifier.

This code corresponds to run 3, a late entry to fix a show-stopping bug in producing the
test results. Run 3 achieved a test F-score of 73.68%, and can be found as our entry on the
[post-evaluation leaderboard](https://competitions.codalab.org/competitions/17326#results)
on CodaLab. The confidence interval of this score overlaps with the high score of 75%.


## Input data

The input data is [available on Zenodo](https://zenodo.org/record/1183358). Download the
Zip file and extract it into `discriminatt/more-data`.


## Reproducing results

To reproduce this result:

- Activate a Python 3 environment where you can install packages

- [Install ConceptNet 5.5](https://github.com/commonsense/conceptnet5/wiki/Build-process).
  Be warned that this comes with a number of setup steps of its own. You won't
  need strictly need the database, but you will at least need its
  `data/db/wiktionary.db` file, for lemmatizing words.

- Run `python setup.py develop`

- Make sure you have the input data in `discriminatt/more-data`, as described above

- Run `python discriminatt/classifier.py`

The output results come from the full classifier, followed by "ablated"
versions of the classifier with features disabled, followed by a simple
one-feature heuristic described in our paper.
