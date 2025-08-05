# src/utils/data.py
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class MyDataset(Dataset):
    def __init__(self, data, covariates, return_indices=False, transform=None):
        """
        Dataset class for brain imaging data and associated covariates
        
        Args:
            data: Brain imaging features (Y in the papers)
            covariates: Subject covariates like age, sex, etc (X in the papers)
            return_indices: If True, also return the index with each sample
            transform: Optional transform to be applied to data
        """
        self.data = data
        self.covariates = covariates
        self.num_subjects = len(data)
        self.transform = transform
        self.return_indices = return_indices

    def __getitem__(self, index):
        features = self.data[index]
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        elif isinstance(features, torch.Tensor):
            features = features.float()
            
        covs = self.covariates[index]
        if isinstance(covs, np.ndarray):
            covs = torch.from_numpy(covs).float()
        elif isinstance(covs, torch.Tensor):
            covs = covs.float()
            
        if self.transform:
            features = self.transform(features)
            
        if self.return_indices:
            return features, covs, index
            
        return features, covs

    def __len__(self):
        return self.num_subjects
        
        
def process_covariates(covariates_df):
    """Process covariates into the format needed by the model."""
    try:
        # Continuous variables
        age_icv = covariates_df[['Age', 'ICV']].values
        
        # Categorical variables
        categorical_cols = ['Sex']
        one_hot_encodings = []
        
        for col in categorical_cols:
            if col not in covariates_df.columns:
                raise KeyError(f"Column '{col}' not found in covariates DataFrame")
            one_hot = pd.get_dummies(covariates_df[col], prefix=col).values
            one_hot_encodings.append(one_hot)
        
        # Combine all covariates
        return np.hstack([age_icv] + one_hot_encodings)
    except KeyError as e:
        raise KeyError(f"Missing required column: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing covariates: {str(e)}")
    
def load_train_data(data_path, val_size, logger):
    """Load and process training and validation data."""
    logger.info("Loading training data...")
    
    train_data = pd.read_csv(data_path / 'train_data.csv')
    train_covariates = pd.read_csv(data_path / 'train_covariates.csv')
    test_data = pd.read_csv(data_path / 'test_data.csv')
    test_covariates = pd.read_csv(data_path / 'test_covariates.csv')
    
    train_covariates_processed = process_covariates(train_covariates)
    test_covariates_processed = process_covariates(test_covariates)
    
    train_data_np = train_data.to_numpy()
    test_data_np = test_data.to_numpy()
    
    # Split training data into train and validation sets
    indices = np.arange(len(train_data_np))
    np.random.shuffle(indices)
    split_idx = int(len(indices) * (1 - val_size))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Create train and validation sets
    train_data_split = train_data_np[train_indices]
    train_cov_split = train_covariates_processed[train_indices]
    val_data_split = train_data_np[val_indices]
    val_cov_split = train_covariates_processed[val_indices]
    
    return (train_data_split, train_cov_split, 
            val_data_split, val_cov_split,
            test_data_np, test_covariates_processed)

def load_test_data(data_path, logger, data_type='test'):
    """Load and process test data."""
    logger.info("Loading test data...")
    
    if data_type == 'test':
        test_data = pd.read_csv(f"{data_path}/test_data.csv")
        test_covariates = pd.read_csv(f"{data_path}/test_covariates.csv")
    elif data_type == 'train':
        test_data = pd.read_csv(f"{data_path}/train_data.csv")
        test_covariates = pd.read_csv(f"{data_path}/train_covariates.csv")
    
    # Process covariates
    processed_covariates = process_covariates(test_covariates)
    
    return test_data, test_covariates, processed_covariates