# src/inference/density_inference.py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from scipy import stats

logger = logging.getLogger(__name__)

def calculate_conditional_density(model, y_observed, x_covariates, feature_cols=None, device='cpu'):
    """
    Calculate p(Y|X) for observed data points directly using the model's log_prob method.
    
    Args:
        model: Trained ConditionalRealNVP model
        y_observed: Observed values tensor or numpy array [batch_size, input_dim]
        x_covariates: Corresponding covariates tensor or numpy array [batch_size, c_dim]
        feature_cols: List of feature column names (optional)
        device: Computing device ('cpu' or 'cuda')
        
    Returns:
        density_df: DataFrame containing log_prob, prob, and normalized_prob values
    """
    if not isinstance(y_observed, torch.Tensor):
        y_tensor = torch.tensor(y_observed, dtype=torch.float32)
    else:
        y_tensor = y_observed.float()
        
    if not isinstance(x_covariates, torch.Tensor):
        x_tensor = torch.tensor(x_covariates, dtype=torch.float32)
    else:
        x_tensor = x_covariates.float()
    
    model = model.to(device)
    y_tensor = y_tensor.to(device)
    x_tensor = x_tensor.to(device)
    
    model.eval()
    
    logger.info(f"Calculating conditional density for {len(y_tensor)} data points...")
    
    with torch.no_grad():
        try:
            log_probs, log_dets, log_prob_base = model.log_prob(y_tensor, x_tensor)
            per_feature_log_probs, z_values = model.per_feature_log_prob(y_tensor, x_tensor)
            
            per_feature_log_probs_np = per_feature_log_probs.cpu().numpy()
            z_values_np = z_values.cpu().numpy()
            
            probs = torch.exp(log_probs)
            
            log_probs_np = log_probs.cpu().numpy()
            probs_np = probs.cpu().numpy()
            
            log_probs_per_dim_np = log_probs_np / y_tensor.shape[1]
            
            density_df = pd.DataFrame({
                'log_prob': log_probs_np,
                'log_prob_per_dim': log_probs_per_dim_np,
                'prob': probs_np
            })
            density_df['prob_rank_normalized'] = density_df['log_prob_per_dim'].rank(pct=True)
            
            # Create per-feature log probability dataframe
            if feature_cols is None:
                feature_cols = [f'feature_{i}' for i in range(per_feature_log_probs_np.shape[1])]
            
            feature_logprob_data = {}
            z_data = {}
            
            for i, col in enumerate(feature_cols):
                feature_logprob_data[col] = per_feature_log_probs_np[:, i]
                z_data[col] = z_values_np[:, i]
                
            feature_logprob_df = pd.DataFrame(feature_logprob_data) 
            z_df = pd.DataFrame(z_data)
            
            logger.info("Conditional density calculation completed successfully")
            return density_df, feature_logprob_df, z_df
            
        except Exception as e:
            logger.error(f"Error calculating conditional density: {str(e)}")
            raise

def compute_density_based_deviation_scores(density_df, feature_cols, results_dir=None):
    """
    Compute deviation scores based on the probability density values.
    Lower probability = higher deviation from the expected distribution.
    
    Args:
        density_df: DataFrame with probability values from calculate_conditional_density
        feature_cols: Names of the feature columns (for reporting)
        results_dir: Directory to save results (optional)
        
    Returns:
        deviation_df: DataFrame with various deviation metrics
    """
    try:
        deviation_df = pd.DataFrame()
        
        # Negative log probability (NLP) scores
        deviation_df['nlp_score'] = -density_df['log_prob_per_dim']
        
        # Probability percentile (0 = most unusual, 1 = most typical)
        deviation_df['density_percentile'] = density_df['prob_rank_normalized']
        
        # Convert to classic "deviation score" where:
        # 0 = perfectly typical
        # >2 = unusual (roughly equivalent to 2 std deviations in normal distribution)
        # >3 = very unusual (roughly equivalent to 3 std deviations)
        deviation_df['deviation_score'] = -stats.norm.ppf(density_df['prob_rank_normalized'].clip(0.001, 0.999))
        
        if results_dir is not None:
            plt.figure(figsize=(10, 6))
            sns.histplot(deviation_df['deviation_score'], kde=True)
            plt.title('Distribution of Density-Based Deviation Scores')
            plt.xlabel('Deviation Score')
            plt.axvline(x=2, color='r', linestyle='--', alpha=0.7, label='>2: Unusual')
            plt.axvline(x=3, color='darkred', linestyle='--', alpha=0.7, label='>3: Very Unusual')
            plt.legend()
            plt.savefig(results_dir / 'deviation_score_distribution.png')
            plt.close()
            
        logger.info("Density-based deviation scores computed successfully")
        return deviation_df
        
    except Exception as e:
        logger.error(f"Error computing density-based deviation scores: {str(e)}")
        raise
