import numpy as np
from scipy.spatial import distance
from scipy.stats import pearsonr

class VectorComparator:
    def __init__(self, vector_a, vector_b):
        self.vector_a = np.array(vector_a)
        self.vector_b = np.array(vector_b)

    def euclidean_distance(self):
        return np.linalg.norm(self.vector_a - self.vector_b)

    def cosine_similarity(self):
        vector_a = self.vector_a.flatten()
        vector_b = self.vector_b.flatten()

        dot_product = np.dot(vector_a.T, vector_b)
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        return dot_product / (norm_a * norm_b)


    def manhattan_distance(self):
        return np.sum(np.abs(self.vector_a - self.vector_b))

    def mahalanobis_distance(self):
        cov_matrix = np.cov(np.stack((self.vector_a, self.vector_b), axis=0))
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        diff = self.vector_a - self.vector_b
        return np.sqrt(np.dot(np.dot(diff, inv_cov_matrix), diff.T))

    def pearson_correlation(self):
        corr, _ = pearsonr(self.vector_a, self.vector_b)
        return corr

    def jaccard_similarity(self):
        intersection = np.sum(np.minimum(self.vector_a, self.vector_b))
        union = np.sum(np.maximum(self.vector_a, self.vector_b))
        return intersection / union