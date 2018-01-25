import sqlite3

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
    "CREATE INDEX phrases_second_lemma ON phrases (second_lemma)"
]


def build_phrases_database(db, filename):
    db.execute("DROP TABLE IF EXISTS phrases")
    with db as _transaction:
        for statement in SCHEMA:
            db.execute(statement)

    with db as _transaction:
        num_lines = sum(1 for line in open(filename))
        for line in tqdm(open(filename), total=num_lines):
            phrase, count = line.lower().split(',')
            first_word, second_word = phrase.split()
            count = int(count)
            add_entry(db, phrase, first_word, second_word, count)


def add_entry(db, phrase, first_word, second_word, count):
    first_lemma = LEMMATIZER.lookup('en', first_word)[0]
    second_lemma = LEMMATIZER.lookup('en', second_word)[0]

    db.execute(
            "INSERT OR IGNORE INTO phrases (phrase, first_word, second_word, count, "
            "first_lemma, second_lemma) VALUES (?, ?, ?, ?, ?, ?)",
            (phrase, first_word, second_word, count, first_lemma, second_lemma)
        )

if __name__ == '__main__':
    filename = get_external_data_filename('google-books-2grams.txt')
    db = sqlite3.connect(get_external_data_filename('phrases.db'))
    build_phrases_database(db, filename)
