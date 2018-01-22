from conceptnet5.nodes import standardized_concept_uri


def wikipedia_connected_conceptnet_nodes(db, start):
    nodes = set()
    c = db.cursor()
    c.execute("SELECT lemma FROM words WHERE page=? LIMIT 1000", (start.casefold(),))
    for row in c.fetchall():
        lemma = row[0]
        nodes.add(standardized_concept_uri('en', lemma))
    return list(nodes)
