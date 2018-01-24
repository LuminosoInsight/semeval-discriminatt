import pathlib
import numpy as np

from conceptnet5.vectors.formats import load_hdf


class StandaloneSMEModel:
    RELEVANT_RELATIONS = [
        '/r/RelatedTo',
        '/r/IsA',
        '/r/HasProperty',
        '/r/PartOf',
    ]
    def __init__(self, dirname):
        """
        Load the files exported from a model trained with ConceptNet's
        implementation of Semantic Matching Energy.

        `terms-similar.h5` is the (|V| x 300) matrix of term embeddings. (It's
        named that because using the embeddings directly, instead of operating
        on them with a relation, is meant to represent the SimilarTo relation.)

        `relations.h5` is the (24 x 10) matrix of relation embeddings.

        `assoc.npy` is a 3-tensor with shape (10 x 300 x 300), which relates
        two term embeddings and a relation embedding. Multiplying two vectors
        by this tensor in the appropriate dimensions gives you a prediction for
        the third vector.
        """
        path = pathlib.Path(dirname)
        self.rel_embeddings = load_hdf(str(path / 'relations.h5'))
        self.term_embeddings = load_hdf(str(path / 'terms-similar.h5'))
        self.assoc_tensor = np.load(str(path / 'assoc.npy'))

    def predict_relations_forward(self, term1, term2):
        """
        Given two terms (which are in ConceptNet URI form and in the
        vocabulary), predict the relations between them (in the direction
        from term1 to term2).
        """
        vec1 = self.term_embeddings.loc[term1]
        vec2 = self.term_embeddings.loc[term2]
        rel_vec = np.einsum(
            'ijk,j,k->i',
            self.assoc_tensor, vec1, vec2
        )
        rels = self.rel_embeddings.dot(rel_vec)
        return rels

    def predict_relations_backward(self, term1, term2):
        """
        Given two terms (which are in ConceptNet URI form and in the
        vocabulary), predict the relations between them (in the direction
        from term2 to term1).
        """
        vec1 = self.term_embeddings.loc[term1]
        vec2 = self.term_embeddings.loc[term2]
        rel_vec = np.einsum(
            'ijk,j,k->i',
            self.assoc_tensor, vec2, vec1
        )
        rels = self.rel_embeddings.dot(rel_vec)
        return rels

    def predict_discriminative_relations(self, term1, term2):
        forward = self.predict_relations_forward(term1, term2)
        backward = self.predict_relations_backward(term1, term2)
        return np.array([
            forward.loc['/r/RelatedTo'] + backward.loc['/r/RelatedTo'],
            forward.loc['/r/IsA'],
            forward.loc['/r/HasProperty'],
            forward.loc['/r/HasA'],
            backward.loc['/r/PartOf'],
            backward.loc['/r/HasProperty']
        ])

    def predict_terms_forward(self, rel, term):
        """
        Given a relation and a term on the left side of it, predict terms
        on the right side of it.
        """
        term1_vec = self.term_embeddings.loc[term]
        rel_vec = self.rel_embeddings.loc[rel]

        term2_vec = np.einsum(
            'ijk,i,j->k',
            self.assoc_tensor, rel_vec, term1_vec
        )
        return self.term_embeddings.dot(term2_vec)

    def predict_terms_backward(self, rel, term):
        """
        Given a relation and a term on the right side of it, predict terms
        on the left side of it.
        """
        term2_vec = self.term_embeddings.loc[term]
        rel_vec = self.rel_embeddings.loc[rel]

        term1_vec = np.einsum(
            'ijk,i,->j',
            self.assoc_tensor, rel_vec, term2_vec
        )
        return self.term_embeddings.dot(term1_vec)

    def __contains__(self, term):
        return term in self.term_embeddings.index

    def num_rels(self):
        return 6
