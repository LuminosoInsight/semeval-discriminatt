
def count_for_phrases(db, word, attribute):
    c = db.cursor()
    c.execute("SELECT count FROM phrases WHERE (first_word=? AND second_word=?) OR (first_word=? "
              "AND second_word=?)", (word, attribute, attribute, word))
    results = c.fetchall()
    return sum([result[0] for result in results])
