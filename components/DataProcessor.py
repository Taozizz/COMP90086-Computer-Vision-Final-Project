import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import csv

class DataProcessor:
    def __init__(self, file_path, test_file_path, save_path, dataset_base_path):
        self.file_path = file_path
        self.test_file_path = test_file_path
        self.save_path = save_path
        self.dataset_base_path = dataset_base_path
        self.train_data = None
        self.all_right_names = None
        self.train_subset = None
        self.val_subset = None
        self.train_dict_simple = None
        self.val_dict_simple = None
        self.test_dict = None

    def load_data(self):
        self.train_data = pd.read_csv(os.path.join(self.file_path))
        self.all_right_names = self.train_data['right'].unique().tolist()

    def split_data(self, test_size=0.2, random_state=42):
        self.train_subset, self.val_subset = train_test_split(
            self.train_data, test_size=test_size, random_state=random_state)

    @staticmethod
    def generate_random_right_with_original(left_name, original_right_name, all_right_names, num=19):
        all_right_names = [name for name in all_right_names if name != original_right_name]
        random_right = np.random.choice(all_right_names, num, replace=False).tolist()
        random_right.insert(0, original_right_name)
        return random_right

    def generate_dictionaries(self):
        self.val_subset['special_right'] = self.val_subset.apply(
            lambda row: self.generate_random_right_with_original(row['left'], row['right'], self.all_right_names), axis=1)
        self.train_dict_simple = dict(zip(
            self.get_full_path(self.train_subset['left'], 'train', 'left'),
            self.get_full_path(self.train_subset['right'], 'train', 'right')
        ))
        self.val_dict_simple = dict(zip(
            self.get_full_path(self.val_subset['left'], 'train', 'left'),
            self.get_full_path(self.val_subset['special_right'].apply(lambda x: x[0]), 'train', 'right')
        ))

    def save_dictionaries(self):
        os.makedirs(self.save_path, exist_ok=True)
        train_dict_file_path = os.path.join(self.save_path, 'train_dict.pkl')
        val_dict_file_path = os.path.join(self.save_path, 'val_dict.pkl')
        with open(train_dict_file_path, 'wb') as f:
            pickle.dump(self.train_dict_simple, f)
        with open(val_dict_file_path, 'wb') as f:
            pickle.dump(self.val_dict_simple, f)
        print(f"Dictionaries saved to '{train_dict_file_path}' and '{val_dict_file_path}'")

    def process_test_candidates(self):
        data_dict = {}
        with open(self.test_file_path, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                key = row["left"]
                values = [row[col] for col in reader.fieldnames if col != "left"]
                data_dict[key] = values
        
        self.test_dict = dict(zip(
            self.get_full_path(pd.Series(data_dict.keys()), 'test', 'left'),
            [self.get_full_path(pd.Series(values), 'test', 'right') for values in data_dict.values()]
        ))
        
        pkl_filepath = os.path.join(self.save_path, 'test_dict.pkl')
        with open(pkl_filepath, 'wb') as pkl_file:
            pickle.dump(self.test_dict, pkl_file)
        print(f"Dictionaries saved to {pkl_filepath}")

    def get_full_path(self, file_names_series, dataset_type, side):
        return self.dataset_base_path + f'/{dataset_type}/{side}/' + file_names_series + '.jpg'
