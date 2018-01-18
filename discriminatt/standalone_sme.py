import pathlib
import numpy as np

from conceptnet5.vectors.formats import load_hdf


class StandaloneSMEModel:
    def __init__(self, dirname):
        path = pathlib.Path(dirname)
        self.rel_embeddings = load_hdf(str(path / 'relations.h5'))
        self.term_embeddings = load_hdf(str(path / 'terms-similar.h5'))
        self.assoc_tensor = np.load(str(path / 'assoc.npy'))

    def predict_relations(self, term1, term2):
        vec1 = self.term_embeddings.loc[term1]
        vec2 = self.term_embeddings.loc[term2]
        rel_vec = np.einsum(
            'ijk,j,k->i',
            self.assoc_tensor, vec1, vec2
        )
        rels = self.rel_embeddings.dot(rel_vec)
        return rels
