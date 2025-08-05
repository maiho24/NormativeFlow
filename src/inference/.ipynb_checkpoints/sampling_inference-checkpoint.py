# src/inference/sampling_inference.py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from scipy import stats
import re

logger = logging.getLogger(__name__)

def generate_mc_samples(model, covariates, saved_batch_dir, num_samples=1000, device='cpu', batch_size=1):
    """
    Generate Monte Carlo samples for each covariate combination using cRealNVP.
    
    Args:
        model: Trained cRealNVP model
        covariates: Processed covariates array
        saved_batch_dir: Directory to save batch results
        num_samples: Number of samples per covariate combination
        device: Computing device ('cpu' or 'cuda')
    
    Returns:
        List of numpy arrays containing samples for each covariate combination
    """
    model = model.to(device)
    model.eval()
    
    saved_batch_dir = Path(saved_batch_dir)
    saved_batch_dir.mkdir(exist_ok=True, parents=True)
    saved_batch_files = []
    
    try:
        if isinstance(covariates, torch.Tensor):
            covariates_tensor = covariates.to(device)
        else:
            covariates_tensor = torch.FloatTensor(covariates).to(device)
        
        total_batches = (len(covariates_tensor) + batch_size - 1) // batch_size
        logger.info(f"Starting MC sampling for {len(covariates_tensor)} covariate combinations "
                    f"in {total_batches} batches")
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(covariates_tensor))
            
            logger.info(f"Processing batch {batch_idx+1}/{total_batches} (indices {start_idx}:{end_idx})")
            
            batch_covariates = covariates_tensor[start_idx:end_idx].to(device)
            
            with torch.no_grad():
                batch_expanded = batch_covariates.repeat_interleave(num_samples, dim=0)
                batch_samples = model.sample(len(batch_expanded), batch_expanded, device).cpu().numpy()
                batch_samples = batch_samples.reshape(len(batch_covariates), num_samples, -1)
            
            batch_filename = saved_batch_dir / f"batch_{batch_idx:04d}.npy"
            np.save(batch_filename, batch_samples)
            saved_batch_files.append(batch_filename)
            
            del batch_covariates, batch_expanded, batch_samples
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        logger.info(f"MC sampling completed successfully. Results saved in {len(saved_batch_files)} files.")
        return saved_batch_files
        
    except Exception as e:
        logger.error(f"MC sampling failed: {str(e)}")
        raise

def compute_distribution_metrics(batch_files, covariates_df, feature_cols, results_dir):
    """
    Compute comprehensive distribution metrics from MC samples.
    
    Args:
        batch_files: List of saved batch files
        covariates_df: Original covariates DataFrame
        feature_cols: List of feature column names
        results_dir: Directory to save results
    """
    try:
        metrics = {
            'means': [],
            'medians': [],
            'stds': [],
            'quantiles_25': [],
            'quantiles_75': [],
            'quantiles_05': [],
            'quantiles_95': [],
            'skewness': [],
            'kurtosis': [],
            'iqrs': [],
            'mads': []
        }
        
        for batch_file in batch_files:
            batch_idx = int(re.search(r'batch_(\d+)', str(batch_file)).group(1))
            logger.info(f"Processing metrics for batch {batch_idx} from file {batch_file}")
            batch_samples = np.load(batch_file)
            
            batch_means = np.mean(batch_samples, axis=1)
            batch_medians = np.median(batch_samples, axis=1)
            
            metrics['means'].append(batch_means)
            metrics['medians'].append(batch_medians)
            metrics['stds'].append(np.std(batch_samples, axis=1))
            metrics['quantiles_25'].append(np.percentile(batch_samples, 25, axis=1))
            metrics['quantiles_75'].append(np.percentile(batch_samples, 75, axis=1))
            metrics['quantiles_05'].append(np.percentile(batch_samples, 5, axis=1))
            metrics['quantiles_95'].append(np.percentile(batch_samples, 95, axis=1))
            metrics['skewness'].append(stats.skew(batch_samples, axis=1))
            metrics['kurtosis'].append(stats.kurtosis(batch_samples, axis=1))
            
            batch_iqrs = metrics['quantiles_75'][-1] - metrics['quantiles_25'][-1]
            batch_mads = np.median(np.abs(batch_samples - batch_medians[:, np.newaxis, :]), axis=1)
            
            metrics['iqrs'].append(batch_iqrs)
            metrics['mads'].append(batch_mads)
            
            del batch_samples
        
        for metric_name in metrics:
            combined_metric = np.concatenate(metrics[metric_name], axis=0)
            df = pd.DataFrame(combined_metric, columns=feature_cols)
            df = pd.concat([covariates_df, df], axis=1)
            df.to_csv(results_dir / f'{metric_name}.csv', index=False)
            
        logger.info("Metrics computation and combination completed")
        return metrics
        
    except Exception as e:
        logger.error(f"Distribution metrics computation failed: {str(e)}")
        raise

def compute_deviation_scores(observed_data, results_dir, feature_cols, method='normal'):
    """
    Compute deviation scores using either normal or robust z-scores.
    
    Args:
        observed_data: Observed data values
        results_dir: Directory where metric CSV files were saved
        feature_cols: List of feature column names
        method: 'normal' for (x-mean)/std or 'robust' for (x-median)/MAD
    
    Returns:
        DataFrame with deviation scores
    """
    try:
        if method not in ['normal', 'robust']:
            raise ValueError("Method must be either 'normal' or 'robust'")
        
        if method == 'normal':
            logger.info("Loading means and standard deviations from saved files")
            means_df = pd.read_csv(results_dir / 'means.csv')
            stds_df = pd.read_csv(results_dir / 'stds.csv')
            
            means = means_df[feature_cols].values
            stds = stds_df[feature_cols].values
            
            logger.info(f"Computing normal z-scores (observed - mean)/std")
            logger.info(f"Observed data shape: {observed_data[feature_cols].shape}")
            logger.info(f"Means shape: {means.shape}, Stds shape: {stds.shape}")
            
            z_scores = (observed_data[feature_cols].values - means) / stds
        else:
            logger.info("Loading medians and MADs from saved files")
            medians_df = pd.read_csv(results_dir / 'medians.csv')
            mads_df = pd.read_csv(results_dir / 'mads.csv')
            
            medians = medians_df[feature_cols].values
            mads = mads_df[feature_cols].values
            
            logger.info(f"Computing robust z-scores (observed - median)/MAD")
            logger.info(f"Observed data shape: {observed_data[feature_cols].shape}")
            logger.info(f"Medians shape: {medians.shape}, MADs shape: {mads.shape}")
            
            z_scores = 0.6745 * (observed_data[feature_cols].values - medians) / (mads + 1e-9)
        
        z_scores_df = pd.DataFrame(z_scores, columns=feature_cols)
        
        return z_scores_df
        
    except Exception as e:
        logger.error(f"Deviation score computation failed: {str(e)}")
        raise
        
def compute_multivariate_percentiles_from_batches(y_observed, batch_files, feature_cols):
    """
    Compute percentiles by comparing observed data against previously generated batch samples.
    This function reuses the batch files created by generate_mc_samples.
    
    Args:
        y_observed: Observed values [batch_size, input_dim]
        batch_files: List of batch file paths created by generate_mc_samples
        feature_cols: List of feature column names
        
    Returns:
        percentile_df: DataFrame with percentile scores (0-100) for each feature
        z_score_df: DataFrame with Z-scores derived from percentiles
        bidir_df: DataFrame with bidirectional deviation scores (0-50)
    """
    try:
        if isinstance(y_observed, torch.Tensor):
            y_observed = y_observed.cpu().numpy()
            
        n_observations = len(y_observed)
        n_features = len(feature_cols)
        
        percentiles = np.zeros((n_observations, n_features))
        
        logger.info(f"Computing percentiles for {n_observations} observations using {len(batch_files)} sorted batch files")
        
        first_batch = np.load(batch_files[0])
        batch_size = first_batch.shape[0]
        num_samples = first_batch.shape[1]
        
        obs_idx = 0
        for batch_idx, batch_file in enumerate(batch_files):
            batch_num = int(re.search(r'batch_(\d+)', str(batch_file)).group(1))
            logger.info(f"Processing batch file {batch_idx+1}/{len(batch_files)}: {batch_file} (batch number {batch_num})")
            
            batch_samples = np.load(batch_file)
            
            current_batch_size = batch_samples.shape[0]
            
            for i in range(current_batch_size):
                if obs_idx >= n_observations:
                    logger.warning(f"More batch samples than observations. Stopping at observation {obs_idx}.")
                    break
                    
                samples_i = batch_samples[i]
                
                for k, feature in enumerate(feature_cols):
                    observed_value = y_observed[obs_idx, k]
                    below_count = np.sum(samples_i[:, k] < observed_value)
                    percentiles[obs_idx, k] = 100 * below_count / num_samples
                
                obs_idx += 1
            
            del batch_samples
            
            if obs_idx >= n_observations:
                break
        
        if obs_idx < n_observations:
            logger.warning(f"Not enough batch samples for all observations. Processed {obs_idx}/{n_observations}.")
        
        percentile_df = pd.DataFrame(percentiles, columns=feature_cols)
        
        z_score_dict = {}
        for col in feature_cols:
            p_scaled = percentile_df[col] / 100
            p_clipped = np.clip(p_scaled, 0.001, 0.999)
            z_score_dict[col] = pd.Series(stats.norm.ppf(p_clipped), index=percentile_df.index)
        z_score_df = pd.DataFrame(z_score_dict)
            
        bidir_scores = abs(percentiles - 50)
        bidir_df = pd.DataFrame(bidir_scores, columns=feature_cols)
            
        logger.info("Percentile-based scores computed successfully")
        return percentile_df, z_score_df, bidir_df
        
    except Exception as e:
        logger.error(f"Error computing percentile-based scores: {str(e)}")
        raise