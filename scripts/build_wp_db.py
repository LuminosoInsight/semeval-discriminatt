import wordfreq
import sqlite3
from tqdm import tqdm
from conceptnet5.language.english import LEMMATIZER
from discriminatt.data import get_external_data_filename

SCHEMA = [
    """
    CREATE TABLE words (
        page TEXT,
        word TEXT,
        lemma TEXT
    )
    """,
    "CREATE INDEX words_page ON words (page)",
    "CREATE INDEX words_word ON words (word)",
    "CREATE INDEX words_lemma ON words (lemma)",
    "CREATE UNIQUE INDEX words_unique on words (page, word)",
]


def build_wp_database(db, filename):
    db.execute("DROP TABLE IF EXISTS words")
    with db as _transaction:
        for statement in SCHEMA:
            db.execute(statement)

    with db as _transaction:
        num_lines = sum(1 for line in open(filename))
        for line in tqdm(open(filename), total=num_lines):
            title, text = line.split('\t', 1)
            words = wordfreq.tokenize(text.rstrip(), 'en')
            for word in words:
                add_entry(db, title, word)


def add_entry(db, title, word):
    lemma = LEMMATIZER.lookup('en', word)[0]
    title = title.lower().split(" (")[0]
    if wordfreq.zipf_frequency(lemma, 'en') < 6 and wordfreq.zipf_frequency(word, 'en') < 6:
        db.execute(
            "INSERT OR IGNORE INTO words (page, word, lemma) VALUES (?, ?, ?)",
            (title, word, lemma)
        )


if __name__ == '__main__':
    filename = get_external_data_filename('en-wp-1word-summaries.txt')
    db = sqlite3.connect(get_external_data_filename('wikipedia-summary.db'))
    build_wp_database(db, filename)
