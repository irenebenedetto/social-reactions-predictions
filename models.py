
from transformers import AutoTokenizer, AutoModel

import torch
import numpy as np
import random
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import transformers
from torch.utils.tensorboard import SummaryWriter
import os
import json
import argparse


def myLoss(outputs, labels):
    # (batch size, 5)
    loss = criterion(outputs, labels)
    return loss

class RobertaRegressorTextOnly(torch.nn.Module):
    def __init__(self, n_regression, model_path = 'xlm-roberta-base'):
        super(RobertaRegressor, self).__init__()
        
        self.fe = AutoModel.from_pretrained(model_path)

        self.regressor = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(768, n_regression),  # 1024 or 768
            torch.nn.ReLU()
        )

    def forward(self, x):
        x = self.fe(**x)['pooler_output']
        x = self.regressor(x)
        return x


class InfluencerDatasetTextOnly(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str,
                 labels_path: str,
                 model_path: str,
                 max_document_length: int = 512):
        
        self.documents = pd.read_csv(dataset_path, index_col='id')['content'].values
        self.labels = pd.read_csv(labels_path, index_col='id').values
        self.max_document_length = max_document_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.documents)

    def vectorize_data(self, document):
        
        x = self.tokenizer(document,
            max_length=self.max_document_length, padding='max_length',
            truncation=True)
        x['input_ids'] = torch.tensor(x['input_ids']).to(device)
        x['attention_mask'] = torch.tensor(x['attention_mask']).to(device)

        return x

    def __getitem__(self, index):
        """Generate one batch of data"""

        text = self.documents[index]
        y = self.labels[index]

        x = self.vectorize_data(text)
        return x, y


class RobertaRegressorWithMetadata(torch.nn.Module):
    def __init__(self, n_regression, model_path = 'xlm-roberta-base'):
        super(RobertaRegressor, self).__init__()
        
        self.fe = AutoModel.from_pretrained(model_path)

        self.regressor = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(768 + 8, n_regression),  # 1024 or 768
            torch.nn.ReLU()
        )

    def forward(self, x, x_str):
        x = self.fe(**x)['pooler_output']
        x = torch.cat([x, x_str], dim=-1)
        x = self.regressor(x)
        return x


class InfluencerDatasetWithMetadata(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str,
                 labels_path: str,
                 model_path: str,
                 max_document_length: int = 512,
                 mean: np.array = None, 
                 std:  np.array = None):
      

        structured_cols = [
            'name',
            'type',
            'subscriberCount',
            'day',
            'year',
            'month',
            'hour',
            'minute'
        ]
        
        data = pd.read_csv(dataset_path, index_col='id')
        self.documents = data['content'].values
        self.structured_info = data[structured_cols].values
        if mean is None and std is None:
            mean, std = self.structured_info.mean(), self.structured_info.std()
            
            
        self.mean = mean 
        self.std = std
        self.structured_info = (self.structured_info - self.mean) / self.std
        self.structured_info = self.structured_info.astype(np.float32) 
        
        self.labels = pd.read_csv(labels_path, index_col='id').values
        self.max_document_length = max_document_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.documents)

    def vectorize_data(self, document):
        
        x = self.tokenizer(document,
            max_length=self.max_document_length, padding='max_length',
            truncation=True)
        x['input_ids'] = torch.tensor(x['input_ids']).to(device)
        x['attention_mask'] = torch.tensor(x['attention_mask']).to(device)

        return x

    def __getitem__(self, index):
        """Generate one batch of data"""

        text = self.documents[index]
        x_str = torch.tensor(self.structured_info[index]).to(device)

        y = self.labels[index]

        x = self.vectorize_data(text)
        return x, x_str, y