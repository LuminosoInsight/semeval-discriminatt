import sqlite3
from tqdm import tqdm
from discriminatt.data import get_external_data_filename

SCHEMA = [
    """
    CREATE TABLE phrases (
        phrase TEXT,
        first_word TEXT,
        second_word TEXT,
        count INTEGER
    )
    """,
    "CREATE INDEX phrases_first_word ON phrases (first_word)",
    "CREATE INDEX phrases_second_word ON phrases (second_word)",
    "CREATE INDEX phrases_count ON phrases (count)"
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
        db.execute(
            "INSERT OR IGNORE INTO phrases (phrase, first_word, second_word, count) VALUES (?, ?, ?, ?)",
            (phrase, first_word, second_word, count)
        )

if __name__ == '__main__':
    filename = get_external_data_filename('google-books-2grams.txt')
    db = sqlite3.connect(get_external_data_filename('phrases.db'))
    build_phrases_database(db, filename)
