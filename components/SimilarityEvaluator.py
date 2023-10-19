import random
import pickle
import os

from components.VectorComparator import VectorComparator

class SimilarityEvaluator:
    
    def __init__(self, model_name, val_set_name, base_dir="feat", metric='cosine'):
        self.model_name = model_name
        self.val_set_name = val_set_name
        self.base_dir = base_dir
        self.metric = metric

    @staticmethod
    def load_saved_features(model_name, set_name, side, base_dir):
        features_file_path = os.path.join(base_dir, model_name, set_name, f"{set_name}_{side}_features.pkl")
        with open(features_file_path, 'rb') as f:
            features_dict = pickle.load(f)
        return features_dict

    @staticmethod
    def load_pkl(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def augment_val_data(self, val_dict, right_features, num_confusing_samples=19):
        all_right_images = list(right_features.keys())
        augmented_val_dict = {}
        for left_img, correct_right_img in val_dict.items():
            confusing_pool = [img for img in all_right_images if img != correct_right_img]
            confusing_samples = random.sample(confusing_pool, num_confusing_samples)
            candidates = [correct_right_img] + confusing_samples
            augmented_val_dict[left_img] = candidates
        return augmented_val_dict

    def load_validation_data(self, val_dict):
        val_left_features = self.load_saved_features(self.model_name, self.val_set_name, "left", self.base_dir)
        val_right_features = self.load_saved_features(self.model_name, self.val_set_name, "right", self.base_dir)
        augmented_val_dict = self.augment_val_data(val_dict, val_right_features)
        return val_left_features, augmented_val_dict, val_right_features 

    def find_top2_similar(self, val_left, candidates_dict, val_candidates_features):
        top2_indices = {}
        for anchor_key, anchor_features in val_left.items():
            similarities = []
            candidates = candidates_dict[anchor_key]
            for candidate_index, candidate_key in enumerate(candidates):
                candidate_features = val_candidates_features[candidate_key]
                
                comparator = VectorComparator(anchor_features, candidate_features)
                similarity = comparator.compute(self.metric)
                
                similarities.append((similarity, candidate_index))

            # Sort the similarities list and get the indices of the top 2 candidates
            similarities.sort(key=lambda x: x[0], reverse=True)
            top2_indices[anchor_key] = [similarities[0][1], similarities[1][1]]
        
        return top2_indices

    def evaluate_accuracy(self, val_dict):
        val_left, augmented_val_dict, val_candidates_features = self.load_validation_data(val_dict)
        top2_indices = self.find_top2_similar(val_left, augmented_val_dict, val_candidates_features)
        count = 0
        for key in top2_indices:
            if top2_indices[key][0] == 0 or top2_indices[key][1] == 0:
                count += 1
        acc = count / len(top2_indices)
        return acc
    
    def get_total_params(self, model):
        total_params = 0
        for layer in model.layers:
            total_params += layer.count_params()
        return total_params
