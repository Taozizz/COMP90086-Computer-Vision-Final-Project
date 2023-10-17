import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics.pairwise import euclidean_distances
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers.experimental import SyncBatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import HeNormal
import numpy as np

from components.VectorComparator import VectorComparator

class SimilarityModel:
    def __init__(self, input_shape):
        self.embedding_model = self._create_embedding_model(input_shape)
        self.triplet_model = self._create_triplet_model(input_shape)

    def _create_embedding_model(self, input_shape):
        base_input = Input(input_shape)
        x = Dense(256, activation='relu')(base_input)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        return Model(base_input, x)

    def _create_triplet_model(self, input_shape):
        def triplet_loss(y_true, y_pred):
            margin = 1
            anchor, positive, negative = y_pred[:, :128], y_pred[:, 128:256], y_pred[:, 256:]
            positive_distance = tf.reduce_sum(tf.square(anchor - positive), axis=1)
            negative_distance = tf.reduce_sum(tf.square(anchor - negative), axis=1)
            return tf.maximum(positive_distance - negative_distance + margin, 0)
        
        anchor_input = Input(input_shape, name='anchor_input')
        positive_input = Input(input_shape, name='positive_input')
        negative_input = Input(input_shape, name='negative_input')
        
        anchor_embedding = self.embedding_model(anchor_input)
        positive_embedding = self.embedding_model(positive_input)
        negative_embedding = self.embedding_model(negative_input)
        
        outputs = tf.keras.layers.Concatenate()([anchor_embedding, positive_embedding, negative_embedding])
        triplet_model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=outputs)
        triplet_model.compile(optimizer=Adam(lr=0.0001), loss=triplet_loss)
        return triplet_model

    def train(self, anchors, positives, negatives):
        self.triplet_model.fit([anchors, positives, negatives], np.zeros(len(anchors)), epochs=50, batch_size=32)

    def compute_similarity(self, feature1, feature2, metric='cosine'):
        comparator = VectorComparator(feature1, feature2)
        if metric == 'cosine':
            return comparator.cosine_similarity()
        elif metric == 'euclidean':
            return -comparator.euclidean_distance()
        elif metric == 'manhattan':
            return -comparator.manhattan_distance()
        elif metric == 'mahalanobis':
            return -comparator.mahalanobis_distance()
        elif metric == 'pearson':
            return comparator.pearson_correlation()
        elif metric == 'jaccard':
            return comparator.jaccard_similarity()
        else:
            raise ValueError(f'Unknown metric: {metric}')

    def predict(self, feature1, feature2, metric='cosine'):
        if feature1.shape != feature2.shape:
            raise ValueError("The shapes of feature1 and feature2 must be the same.")

        similarity_score = self.compute_similarity(feature1, feature2, metric)
        normalized_score = (similarity_score + 1) / 2
        return normalized_score

