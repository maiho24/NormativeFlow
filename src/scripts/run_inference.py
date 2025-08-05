# src/scripts/run_inference.py
import argparse
import sys
import yaml
from pathlib import Path

def create_parser():
    """Create argument parser with detailed help messages."""
    parser = argparse.ArgumentParser(
        description="""
        Brain Normative cRealNVP Model Inference
        
        This script provides multiple methods for analyzing brain imaging data using a trained 
        conditional RealNVP model. You can select which analyses to perform.
        
        AVAILABLE ANALYSIS METHODS:
        
        1. EXACT DENSITY CALCULATION:
           - Calculate exact conditional density directly from the model (how likely an observation is under the model)
           - Flag: --exact_density
           - Output: conditional_density.csv, feature_logprob.csv
        
        2. DISTRIBUTION METRICS:
           - Compute summary statistics via Monte Carlo sampling
           - Flag: --distribution_metrics
           - Requires: --num_samples
           - Outputs: means.csv, medians.csv, stds.csv, iqrs.csv, etc.
        
        3. DEVIATION SCORING METHODS:
           a) Traditional Z-scores:
              - Normal z-scores: (x-mean)/std
              - Robust z-scores: (x-median)/MAD
              - Flag: --zscore_method [normal|robust|both]
              - Requires: --distribution_metrics or pre-existing metric files
              - Output: normal_zscore.csv|robust_zscores.csv
           
           b) Percentile-based scores:
              - Where observed values fall in predicted distribution
              - Flag: --percentile_scores
              - Requires: --num_samples
              - Output: percentile_CDF.csv|percentile_zscores.csv
        
        Example usage:
          # Basic density calculation and scoring
          brain-nvp-inference --model_path models/model.pkl --data_dir data/ --output_dir results/ \\
                              --exact_density --percentile_scores
          
          # Traditional analysis with distribution metrics
          brain-nvp-inference --model_path models/model.pkl --data_dir data/ --output_dir results/ \\
                              --distribution_metrics --num_samples 1000 --zscore_method both
          
          # Comprehensive analysis
          brain-nvp-inference --model_path models/model.pkl --data_dir data/ --output_dir results/ \\
                              --exact_density --distribution_metrics \\
                              --num_samples 1000 --zscore_method both --percentile_scores
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model_path', 
        type=str,
        required=True,
        help='Path to the trained model (required)'
    )
    
    parser.add_argument(
        '--data_dir', 
        type=str,
        required=True,
        help='Directory containing test data files (required)'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str,
        required=True,
        help='Directory for output files (required)'
    )
    
    parser.add_argument(
        '--gpu', 
        action='store_true',
        help='Use GPU for inference if available'
    )
    
    parser.add_argument(
        '--exact_density', 
        action='store_true',
        help='Calculate exact conditional density (overall and feature-wise)'
    )

    parser.add_argument(
        '--percentile_scores', 
        action='store_true',
        help='Compute percentile-based deviation scores (requires --distribution_metrics or pre-existing MC sampling results)'
    )
    
    parser.add_argument(
        '--distribution_metrics', 
        action='store_true',
        help='Compute distribution metrics (mean, median, std, IQR, etc.) via Monte Carlo sampling'
    )
    
    parser.add_argument(
        '--num_samples', 
        type=int, 
        default=1000,
        help='Number of MC samples per covariate combination (required for --distribution_metrics)'
    )
    
    parser.add_argument(
        '--zscore_method',
        type=str,
        choices=['normal', 'robust', 'both'],
        default=None,
        help='Method for computing traditional deviation scores (requires --distribution_metrics or pre-existing metric files)'
    )
    
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=1,
        help='Batch Size for preventing memory overflow during MC sampling'
    )
    
    parser.add_argument(
        '--force_recompute',
        action='store_true',
        help='Force recomputation even if results already exist'
    )
    
    parser.add_argument(
        '--use_train_data',
        action='store_true',
        help='Use train data instead of test data for inference'
    )
    
    return parser

def run_inference(args, config):
    """Run inference using trained cRealNVP model."""
    import torch
    import logging
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from utils import load_test_data, setup_logging
    
    if any([args.exact_density, args.percentile_scores]):
        try:
            from inference import (
                calculate_conditional_density,
                compute_multivariate_percentiles_from_batches
            )
        except ImportError:
            logging.error("Could not import density-based methods.")
            raise
    
    if any([args.distribution_metrics, args.zscore_method]):
        try:
            from inference import (
                generate_mc_samples,
                compute_distribution_metrics,
                compute_deviation_scores
            )
        except ImportError:
            logging.error("Could not import distribution-metric-based methods.")
            raise
    
    # Validate arguments for operations that require fresh computation
    if args.distribution_metrics and args.num_samples is None:
        raise ValueError("--num_samples must be specified when using --distribution_metrics")
        
    if args.percentile_scores and args.num_samples is None:
        raise ValueError("--percentile_scores requires --num_samples")
    
    output_dir = Path(config['paths']['output_dir'])
    if args.use_train_data:
        results_dir = output_dir / 'results_train_data'
    else:
        results_dir = output_dir / 'results'
    for directory in [output_dir, results_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir, 'inference')
    
    config_file = results_dir / 'inference_config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    logger.info(f"Saved configuration to {config_file}")
    
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
        
    try:
        logger.info("Loading model...")
        model = torch.load(args.model_path, map_location=torch.device('cpu'))
        model = model.to(device)
        model.eval()
        
        if args.use_train_data:
            test_data, test_covariates_raw, test_covariates_processed = load_test_data(
                config['paths']['data_dir'], 
                logger,
                'train'
            )
        else:
            test_data, test_covariates_raw, test_covariates_processed = load_test_data(
                config['paths']['data_dir'], 
                logger,
                'test'
            )
        
        feature_cols = test_data.columns.tolist()
        saved_batch_files = None
        
        # ===== Density-based approaches ===== #
        density_file = results_dir / 'conditional_density.csv'
        feature_logprob_file = results_dir / 'feature_logprob.csv'
        density_df = None
        feature_logprob_df = None
        
        # Calculate or load exact conditional density
        if args.exact_density:
            if density_file.exists() and not args.force_recompute:
                logger.info("Loading existing density...")
                density_df = pd.read_csv(density_file)
            elif not density_file.exists():
                logger.info("Calculating exact density...")    
                density_df, feature_logprob_df, _ = calculate_conditional_density(
                    model=model,
                    y_observed=test_data[feature_cols].values,
                    x_covariates=test_covariates_processed,
                    feature_cols=feature_cols,
                    device=device
                )
                density_df.to_csv(density_file, index=False)
                feature_logprob_df.to_csv(feature_logprob_file, index=False)
                
                logger.info("Exact density computed successfully.")
        
        # ===== Distribution metrics via Monte Carlo sampling ===== #
        metrics_files = {
            'means': results_dir / 'means.csv',
            'medians': results_dir / 'medians.csv', 
            'stds': results_dir / 'stds.csv',
            'mads': results_dir / 'mads.csv'
        }
        
        metrics_exist = all(file.exists() for file in metrics_files.values())
        need_metrics = args.distribution_metrics or args.zscore_method
        metrics = None
        
        # Try to load existing metrics if they exist, we need them, and not forcing recomputation
        if need_metrics and metrics_exist and not args.force_recompute:
            logger.info("Loading existing distribution metrics...")
            try:
                test_cov_cols = len(test_covariates_raw.columns)
                metrics = {}
                for name, file_path in metrics_files.items():
                    df = pd.read_csv(file_path)
                    metrics[name] = df.iloc[:, test_cov_cols:].values
                logger.info("Distribution metrics loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load metrics: {str(e)}. Will recompute if needed.")
                metrics = None
        
        # Decide whether to compute metrics (if they're needed but couldn't be loaded)
        compute_metrics = args.distribution_metrics or (
            args.zscore_method and metrics is None and need_metrics
        )
        
        # If we need to compute metrics (either explicitly requested or needed for z-scores)
        if compute_metrics:
            if args.num_samples is None:
                logger.error("--num_samples is required to compute distribution metrics")
                raise ValueError("--num_samples is required when computing distribution metrics")
                
            # Generate MC samples
            logger.info(f"Generating {args.num_samples} Monte Carlo samples per covariate combination...")
            saved_batch_files = generate_mc_samples(
                model=model,
                covariates=test_covariates_processed,
                saved_batch_dir=results_dir / 'sampling_results',
                num_samples=args.num_samples,
                device=device,
                batch_size=args.batch_size,
            )
            
            # Compute distribution metrics
            logger.info("Computing distribution metrics...")
            metrics = compute_distribution_metrics(
                batch_files=saved_batch_files,
                covariates_df=test_covariates_raw,
                feature_cols=feature_cols,
                results_dir=results_dir
            )
            
            logger.info("Distribution metrics computed successfully.")
        
        # ===== Distribution-based Z-score computation ===== #
        normal_zscore_file = results_dir / 'normal_zscores.csv'
        robust_zscore_file = results_dir / 'robust_zscores.csv'
        
        if args.zscore_method:
            if metrics is None:
                logger.error("Failed to load or compute distribution metrics")
                raise ValueError("Distribution metrics could not be loaded or computed")
            
            if args.zscore_method in ['normal', 'both']:
                if normal_zscore_file.exists() and not args.force_recompute:
                    logger.info(f"File existed at {normal_zscore_file}")
                else:
                    logger.info("Computing normal z-scores...")
                    normal_zscores = compute_deviation_scores(
                        observed_data=test_data,
                        results_dir=results_dir,
                        feature_cols=feature_cols,
                        method='normal'
                    )
                    normal_zscores.to_csv(normal_zscore_file, index=False)
                    logger.info("Normal z-scores computed successfully.")
            
            if args.zscore_method in ['robust', 'both']:
                if robust_zscore_file.exists() and not args.force_recompute:
                    logger.info(f"File existed at {robust_zscore_file}")
                else:
                    logger.info("Computing robust z-scores...")
                    robust_zscores = compute_deviation_scores(
                        observed_data=test_data,
                        results_dir=results_dir,
                        feature_cols=feature_cols,
                        method='robust'
                    )
                    robust_zscores.to_csv(robust_zscore_file, index=False)
                    logger.info("Robust z-scores computed successfully.")
    
        # ===== Percentile-based approach ===== #
        percentile_file = results_dir / 'percentile_CDF.csv'
        percentile_z_file = results_dir / 'percentile_zscores.csv'
        percentile_df = None
        percentile_z_df = None
        sampling_dir = results_dir / 'sampling_results'
        
        if args.percentile_scores:
            if (saved_batch_files is None) and sampling_dir.exists():
                existing_batch_files = list(sampling_dir.glob("batch_*.npy"))
                if existing_batch_files:
                    logger.info(f"Found {len(existing_batch_files)} existing batch files.")
                    saved_batch_files = sorted(existing_batch_files)
                
            if saved_batch_files is None:
                logger.info("Sampling results not found. --distribution_metrics is required")
                logger.error("Cannot compute percentile scores without MC samples.")
            else:
                if percentile_file.exists() and not args.force_recompute:
                    logger.info("Loading existing percentile-based scores...")
                    percentile_df = pd.read_csv(percentile_file)
                    logger.info("Percentiles-based scores loaded successfully.")
                else:
                    logger.info("Computing percentile-based scores...")
                    
                    percentile_df, percentile_z_df, _ = compute_multivariate_percentiles_from_batches(
                        y_observed=test_data[feature_cols].values,
                        batch_files=saved_batch_files,
                        feature_cols=feature_cols
                    )
                    percentile_df.to_csv(percentile_file, index=False)
                    percentile_z_df.to_csv(percentile_z_file, index=False)
            
                logger.info("Percentiles-based deviation computation completed successfully.")
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise

def main():
    parser = create_parser()
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    
    config = {
        'paths': {
            'data_dir': args.data_dir,
            'output_dir': args.output_dir,
            'model_path': args.model_path
        },
        'inference': {
            'num_samples': args.num_samples
        },
        'device': {
            'gpu': args.gpu
        },
        'analysis': {
            'exact_density': args.exact_density,
            'distribution_metrics': args.distribution_metrics,
            'zscore_method': args.zscore_method,
            'percentile_scores': args.percentile_scores
        }
    }
    
    run_inference(args, config)

if __name__ == '__main__':
    main()