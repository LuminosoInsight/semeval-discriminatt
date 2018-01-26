def phrase_count(db, lemma_word, lemma_attribute):
    c = db.cursor()
    c.execute("SELECT count FROM phrases WHERE (first_lemma=? AND second_lemma=?) OR (first_lemma=? "
              "AND second_lemma=?)", (lemma_word, lemma_attribute, lemma_attribute, lemma_word))
    results = c.fetchall()
    return sum([result[0] for result in results])


def word_count(db, lemma_word):
    c = db.cursor()
    c.execute(
        "SELECT count FROM words WHERE lemma=?",
        (lemma_word,)
    )
    results = c.fetchall()
    return sum([result[0] for result in results])


def phrase_weight(db, lemma_word, lemma_attribute):
    return (
        (phrase_count(db, lemma_word, lemma_attribute) + 1) /
        (word_count(db, lemma_word) + 1) /
        (word_count(db, lemma_attribute) + 1)
    )

