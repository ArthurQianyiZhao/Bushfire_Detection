"""
@author: Hongzuo Xu
@comments: testbed for time series anomaly detection on synthetic data
(An experiment on the generalization ability to different types of time series anomalies)
"""

import os
import numpy as np
import pandas as pd
import argparse
from src.algorithms.couta_algo import COUTA


parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default=f'point')
args = parser.parse_args()

# read data

train_df = pd.read_csv('data_processed/blue_range_bmereading_3.csv')
train_df = train_df.iloc[:, [5, 6]]
print(train_df)
train_df = train_df.head(40000)


res_dir = '@results_showcase/'
os.makedirs(res_dir, exist_ok=True)

path = 'saved_models/bushfire_detection_pattern.pth'

model_configs = {
    'sequence_length': 50,
    'stride': 1,
    'lr': 0.001,
    'num_epochs': 40,
    'kernel_size': 2,
    'hidden_dims': [16],
    'emb_dim': 16,
    'rep_hidden': 16,    #64
    'dropout': 0.0,
    'alpha': 0.1,
    'bias': 1,
    'neg_batch_ratio': 0.2,
    'train_val_pc': 0,
    'es': 0,
    'seed': 24,
    'ss_type': 'full',
    'save_model_path': path
}

model = COUTA(**model_configs)
model.fit(train_df)

# scores = model.predict(test_df)['score_t']
# np.save(res_dir + f'showcase_{args.type}_score_couta.npy', scores)


