import sqlite3
import gzip

from conceptnet5.language.english import LEMMATIZER
from tqdm import tqdm

from discriminatt.data import get_external_data_filename

SCHEMA = [
    """
    CREATE TABLE phrases (
        phrase TEXT,
        first_word TEXT,
        second_word TEXT,
        count INTEGER,
        first_lemma TEXT,
        second_lemma TEXT
    )
    """,
    "CREATE INDEX phrases_first_word ON phrases (first_word)",
    "CREATE INDEX phrases_second_word ON phrases (second_word)",
    "CREATE INDEX phrases_count ON phrases (count)",
    "CREATE INDEX phrases_first_lemma ON phrases (first_lemma)",
    "CREATE INDEX phrases_second_lemma ON phrases (second_lemma)",
    """
    CREATE TABLE words (
        word TEXT,
        count INTEGER,
        lemma TEXT
    )
    """,
    "CREATE INDEX words_word ON words (word)",
    "CREATE INDEX words_count ON words (count)",
    "CREATE INDEX words_lemma ON words (lemma)",
]


def build_phrases_database(db, filename_1grams, filename_2grams):
    db.execute("DROP TABLE IF EXISTS phrases")
    db.execute("DROP TABLE IF EXISTS words")
    with db as _transaction:
        for statement in SCHEMA:
            db.execute(statement)

    with db as _transaction:
        for line in tqdm(gzip.open(filename_1grams, 'rt'), desc='1grams'):
            word, count = line.lower().split('\t')
            count = int(count)
            add_word(db, word, count)

        for line in tqdm(gzip.open(filename_2grams, 'rt'), desc='2grams'):
            phrase, count = line.lower().split('\t')
            if ' ' in phrase:
                first_word, second_word = phrase.split(' ')
                count = int(count)
                add_phrase(db, phrase, first_word, second_word, count)


def add_word(db, word, count):
    lemma = LEMMATIZER.lookup('en', word)[0]

    db.execute(
        "INSERT OR IGNORE INTO words (word, count, lemma) "
        "VALUES (?, ?, ?)",
        (word, count, lemma)
    )


def add_phrase(db, phrase, first_word, second_word, count):
    first_lemma = LEMMATIZER.lookup('en', first_word)[0]
    second_lemma = LEMMATIZER.lookup('en', second_word)[0]

    db.execute(
        "INSERT OR IGNORE INTO phrases (phrase, first_word, second_word, count, "
        "first_lemma, second_lemma) VALUES (?, ?, ?, ?, ?, ?)",
        (phrase, first_word, second_word, count, first_lemma, second_lemma)
    )


if __name__ == '__main__':
    filename_1grams = get_external_data_filename('google-books-1grams.txt.gz')
    filename_2grams = get_external_data_filename('google-books-2grams-more.txt.gz')
    db = sqlite3.connect(get_external_data_filename('phrases.db'))
    build_phrases_database(db, filename_1grams, filename_2grams)
